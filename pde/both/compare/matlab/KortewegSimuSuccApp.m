% The objective is to simulate a system of the form:
% u_t(x,t)=u_x(x,t)+ lambda(x) u(x,t) +gbar(x) u(0,t) + \int_0^x fbar(x,y)u(y,t)dy 
% with the control U(t) on u(1,t)


clc
clear all
close all

%% Parameters 

%Simulation parameters

dx=0.01;
dt=0.01; %Attention : they have to be integer-multiples of one another
T=15; %length of simulation

x=(0:dx:1);
t=(0:dt:T);
xExt=(-1:dx:1);
tExt=(-1:dt:T);
uFinal = zeros(20, 1501, 101);
uFinalBoundry = zeros(20, 1501);
Nt=T/dt+1;
Nx=1/dx+1;
p=max(dx,dt);
times = zeros(Nt, 1);
for sampleNum=1:20
    gainTheta=ones(1,Nx); %for the update law
    %System parameters 
    a=1;
    eps=1;
    loadGamma = getfield(load('../outDelta'), 'deltaArr');
    gamma=loadGamma(sampleNum);
    disp(gamma);
    b=sqrt(a/eps);
        
    gbar =3*cos(gamma*acos(x));
    
        
    lambda=0*x;
    
    fbar=zeros(Nx,Nx);
    for i=1:Nx
        for j=1:i
            fbar(i,j)=cosh(b*(i-j)*dx);
        end
    end
    fbar=fbar*gamma*b^2;
        
    %Known parameters
    Mgbar=gamma*b*sinh(b);
    Mfbar=gamma*b^2*cosh(b);
    rho=1;
    
    %Deduced parameters
    M=Mfbar*rho*exp(Mfbar*rho)*(1+Mgbar*rho)+Mgbar*rho;
    
    g=zeros(1,Nx);
    for i=2:Nx
        g(i)=exp(dx*sum(lambda(1:i-1)));
    end
    g=g.*gbar;
    
    f=zeros(Nx,Nx);
    for i=1:Nx
        for j=1:i
            f(i,j)=exp(dx*sum(lambda(j:i-1)));
        end
    end
    f=f.*fbar;
    
    %Initial conditions
    phi0=ones(1,Nx);
    psi0=zeros(1,Nx);
    
    u=zeros(Nt,Nx); 
    thetaH=zeros(Nt,Nx); 
    initMatrix = getfield(load('../outXInit.mat'), 'xinit');
    u(1, :) = initMatrix(sampleNum, :);
    disp(initMatrix(sampleNum, 1));
    thetaH(1,:)=0;
    Y=zeros(Nt+Nx-1,1); %t>=-1
    Y(1:Nx)=phi0'; % for -1<=t<=0, pace=dx, then pace=dt
    U=zeros(Nt+Nx-1,1); %t>=-1
    U(1:Nx)=psi0';
    thetaHatArr = getfield(load('../estDEEP.mat'), 'out');
    thetaHatArr = reshape(thetaHatArr(sampleNum, :, :), 1501, 101);
    gainArr = getfield(load('../outDEEP.mat'), 'out');
    gainArr = reshape(gainArr(sampleNum, :, :), 1501, 101);
    %% Simulation
    
    closedloop=1; % 0 for open-loop, 1 for closed-loop 
    adaptive=1; % 0 for non-adaptive scheme, 1 for adaptive scheme
    
    theta=zeros(1,Nx);
    
    %Finite differences matrices for the u-PDE
    
    Adiff=zeros(Nx-1,Nx);
    Abound=zeros(Nx-1,Nx);
    Alambda=zeros(Nx-1,Nx);
    Af=zeros(Nx-1,Nx-1);
    
    for i=1:Nx-1
        Adiff(i,[i i+1])=[-1/dx 1/dx];
        Abound(i,1)=gbar(i);
        Alambda(i,i)=lambda(i);
        
        if i>1
            Af(i,:)=[dx/2 dx*ones(1,i-2) dx/2 zeros(1, Nx-1-i)].*fbar(i,1:end-1);
        else
            Af(1,:)=zeros(1,Nx-1);
        end
        
        for j=1:Nx
            if isnan(Adiff(i,j))
                Adiff(i,j)=0;
            end
            if (j<Nx && isnan(Af(i,j)))
                Af(i,j)=0;
            end
        end
    end
    
    % In the non-adaptive case, f and g are known and we calculate q, and then
    % theta
    
    if (adaptive==0) 
    
    %     F0=zeros(Nx,Nx);
    %     for i=1:Nx-1
    %         for j=1:i
    %             F0(i,j)=dx*sum(diag(f(i:Nx-1,j:(Nx-1-(i-j)))));
    %         end
    %     end
    %     
    %     q=F0;
    %     Fq=zeros(Nx,Nx);
    %     J=zeros(1,Nx);
    %     while 1
    %         qprev=q;
    %         for i=1:Nx
    %             for j=1:i
    %                 for k=i:Nx
    %                     J(k)=dx*sum(qprev(k-(i-j)+1:k,k+j-i).*f(k,k-(i-j)+1:k)');
    %                 end
    %                 Fq(i,j)=-dx*sum(J(i:Nx-1));
    %             end
    %         end
    %         q=F0+Fq;
    %         display(norm(q-qprev))
    %         if norm(q-qprev)<0.00000000001 
    %             break;
    %         end
    %     end
    
         load('qEps1Fred');
    %     load('q_succApp_Fred');
      q=q_result;
        
        for i=2:Nx
            %theta(i)=-dx*sum(q(i,1:i-1).*g(1:i-1));
            theta(i)=-simps(x(1:i)',q(i,1:i)'.*g(1:i)');
        end
        theta=theta+q(:,1)'+g; 
    %load('thetaAdaptive');
    end
    
    proj=zeros(1,Nx);
    kappa=zeros(1,Nx);
    
    I=zeros(1,Nx);
    
    ehat0=zeros(1,Nt);
    
    for k=2:Nt
        
        if (closedloop)
            
            tt=(k-2)*dt; % previous time 
        
            %%%%%%%%%%%%%   Computation of theta  %%%%%%%%%%%%%%%%%%%% (only in the adaptive case)
            
            if (adaptive)
%                 now1 = tic;
%                 if (tt<1)
%                     n1=round((Nx-1)*(tt-1)+Nx); % index of (tt -1) in Y and U
%                     n=Nx+k-2; % index of (tt) in Y and U
%                     nt=round(tt/dx+1); % index of (tt) in theta
%                     ehat0(k)=Y(n)-U(n1)-dx*sum(thetaH(k-1,Nx:-1:nt+1).*Y(n1:Nx-1)')-p*sum(thetaH(k-1,nt:-p/dx:1).*Y(Nx:p/dt:n)');
%                     alpha=1+dx*sum(Y(n1:Nx-1).^2)+dt*sum(Y(Nx:n).^2);
%                 else
%                     n1=round(Nx+(tt-1)/dt);
%                     n=Nx+k-2;
%                     ehat0(k)=Y(n)-U(n1)-p*sum(thetaH(k-1,Nx:-p/dx:1).*Y(n1:p/dt:n)');
%                     alpha=1+dt*sum(Y(n1:n).^2);
%                 end
%     
%                 for j=1:Nx
%                     tx=tt-(j-1)*dx; % tt-x
%                     if  tx>= 0 
%                         y=Y(round(Nx+tx/dt));
%                     else
%                         y=Y(round(Nx+tx/dx));
%                     end
%     
%                     if (abs(thetaH(k-1,j))>=M && y*ehat0(k)*thetaH(k-1,j) >0)
%                         proj(j)=0;
%                     else proj(j)=y*ehat0(k);
%                     end
%                 end
%     
%                 dtheta=1/alpha*gainTheta.*proj;
                thetaH(k,:)=thetaHatArr(k, :);
                %thetaH(k,:)=thetaH(k-1,:)+ dt*dtheta;
                %theta=thetaH(k,:);
                theta=thetaHatArr(k, :);    
            end
        
            %%%%%%%%%%%%%%%%%%% Computation of kappa %%%%%%%%%%%%%%%%%%%%%%%
%             kappa(1)=-theta(1);
%             for j=2:Nx
%                 kappa(j)=dx*sum(theta(j:-1:2).*kappa(1:j-1))-theta(j);
%              
%                 kappa(j)=simps(x(1:j)',theta(j:-1:1)'.*kappa(1:j)')-theta(j);
%             
%             end
%             elapsed = toc(now1);
            %times(k) = elapsed;
            kappa(1:end) = gainArr(k, 1:end);
                
        
            %%%%%%%%%%%%%%%%%%%  Computation of U %%%%%%%%%%%%%%%%%%%%%%%
            I(1)=0;
            for j=2:Nx
                nttau=Nx-j+1;
                I(j)= simps(x(nttau:Nx)',kappa(nttau:Nx)'.*theta(Nx:-1:nttau)');
            end
            
            if (tt<1)
                n0=round(Nx-tt*(Nx-1)); % index of 0 in I
                n1=round((Nx-1)*(tt-1)+Nx); % index of (tt -1) in Y and U
                n=Nx+k-2; % index of (tt) in Y and U
                nt=round(tt/dx+1); % index of (tt) in theta
                if tt>0
                    U(Nx+k-1)=simps(xExt(n1:Nx)',kappa(Nx:-1:nt)'.*U(n1:Nx))+simps(x(1:p/dx:nt)',kappa(nt:-p/dx:1)'.*U(Nx:p/dt:n))+simps(xExt(n1:Nx)',I(1:n0)'.*Y(n1:Nx))+simps(x(1:p/dx:nt)',I(n0:p/dx:Nx)'.*Y(Nx:p/dt:n));
                else
                    U(Nx+k-1)=simps(xExt(n1:Nx)',kappa(Nx:-1:nt)'.*U(n1:Nx))+simps(xExt(n1:Nx)',I(1:n0)'.*Y(n1:Nx));
                end
            else
                n1=round(Nx+(tt-1)/dt);
                n=Nx+k-2;
                nt1=round((tt-1)/dt+1);
                U(Nx+k-1)=simps(t(nt1:p/dt:k-1)',kappa(Nx:-p/dx:1)'.*U(n1:p/dt:n))+simps(t(nt1:p/dt:k-1)',I(1:p/dx:Nx)'.*Y(n1:p/dt:n));
            end
            
        end
        
        
        %%%%%%%%%%%%%%%%%%%  Computation of u %%%%%%%%%%%%%%%%%%%%%%%%%%
         
        %dudt=((Adiff+Abound+Alambda)*u(k-1,:)'+Af*u(k-1,1:end-1)')*dt;
        dudt = ((Adiff+Abound+Alambda)*u(k-1, :)'*dt);
        u(k,1:end-1)=u(k-1,1:end-1)+dudt';
        u(k,end)=closedloop*rho*U(Nx+k-1); % Boundary control
        Y(Nx+k-1)=u(k,1);
    
        %%%%%%%%%%%%%%%%%%% Computation of SubSample Savings %%%%%%%%%%%%%
        
    uBoundry = U(Nx:end);
    uFinal(sampleNum, 1:end, 1:end) = u(1:end, 1:end);
    uFinalBoundry(sampleNum, 1:end) = uBoundry(1:end);
    end

    disp(sampleNum);
end
save('testingData/deepu.mat', "uFinal");
save('testingData/deepControl.mat', "uFinalBoundry");
disp(sum(times));
%% Figures

pstep=5; % Step for plotting
tstep=5; % Step for plotting

if closedloop==1

subplot(2,2,1)
mesh(x(1:pstep:end),t(1:tstep:end),u(1:tstep:end,1:pstep:end),'edgecolor','black'); 
view(83,10);
xlabel('x', 'FontName', 'Arial', 'FontSize',18)
ylabel('Time [s]', 'FontName', 'Arial', 'FontSize',18)
zlabel('u(x,t)', 'FontName', 'Arial', 'FontSize',18)
set(gca,'FontName', 'Arial', 'FontSize',18)

subplot(2,2,2)
if adaptive
    mesh(x(1:pstep:end),t(1:tstep:end),thetaH(1:tstep:end,1:pstep:end),'edgecolor','black'); 
    view(83,10);
    xlabel('x', 'FontName', 'Arial', 'FontSize',18)
    ylabel('Time [s]', 'FontName', 'Arial', 'FontSize',18)
    zlabel('thetaH(x,t)', 'FontName', 'Arial', 'FontSize',18)
    set(gca,'FontName', 'Arial', 'FontSize',18)
else
   plot(x,theta, 'color', 'black');
   ylabel('theta(x)', 'FontName', 'Arial', 'FontSize',18)
   xlabel('x', 'FontName', 'Arial', 'FontSize',18) 
end

subplot(2,2,3)
plot(t,U(Nx:end), 'color', 'black');
ylabel('U(t)', 'FontName', 'Arial', 'FontSize',18)
xlabel('Time [s]', 'FontName', 'Arial', 'FontSize',18)

else
    mesh(x(1:pstep:end),t(1:tstep:end),u(1:tstep:end,1:pstep:end),'edgecolor','black');
    view(83,10);
    xlabel('x', 'FontName', 'Arial', 'FontSize',18)
    ylabel('Time [s]', 'FontName', 'Arial', 'FontSize',18)
    zlabel('u(x,t)', 'FontName', 'Arial', 'FontSize',18)
    set(gca,'FontName', 'Arial', 'FontSize',18)

end


    



