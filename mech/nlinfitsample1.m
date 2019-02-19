S = load('reaction');
X = S.reactants;
y = S.rate;
beta0 = S.beta;

beta = nlinfit(X,y,@hougen,beta0)