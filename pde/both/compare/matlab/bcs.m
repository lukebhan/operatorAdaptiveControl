function [uL,uR]=bcs(T,uL,uR)

    global Ucourant closedloop;
    uR=Ucourant*closedloop;
    
end