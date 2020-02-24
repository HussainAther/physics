% Potential Wells for solving time independent Schrodinger Equation
% You change the Well parameters in the code for each well type
% Energies values are in eV (electron volts)
% Lengths are in nm (nanometers)
% Other units S.I.
% xMin and xMax define range for x-axis
% x1, x2, ...  well parameters - distance (nm)
% U1, U2, ...  well parameters = potential energy (eV)

clear all
close all
clc

num = 801;             % Number of data points (odd number)

% Constants ------------------------------------------------------------
hbar = 1.055e-34;      % J.s
e = 1.602e-19;         % C
me = 9.109e-31;        % kg  electron mass
mp = 1.67252e-27;      % kg  proton mass
mn = 1.67482e-27;        % kg neutron mass
eps0 = 8.854e-12;      % F/m

m = me;                % Mass of particle


Ese = 1.6e-19;                      % Energy scaling factor  
Lse = 1e-9;                         % Length scaling factor
Cse = -hbar^2/(2*m) / (Lse^2*Ese);  % Schrodinger Eq constant       

% Potential well parameters --------------------------------------------
U = zeros(num,1);
U_matrix = zeros(num-2);

% Potential well types -----------------------------------------------
disp('  ');
disp('  ');
disp('   POTENTIAL WELL TYPES - enter 1 or 2 or 3 ... or 7');
disp('  ');
disp('   1: Square well');
disp(' ');
disp('   2: Stepped well (Asymmetrical well)');
disp(' ');
disp('   3: Double well');
disp(' ');
disp('   4: Sloping well');
disp(' ');
disp('   5: Truncated Parabolic well');
disp(' ');
disp('   6: Morse Potential');
disp(' ');
disp('   7: Parabolic fit to Morse Potential');
disp(' ');
disp('   8: Lattice');
disp(' ');
wellType = input('Specify well type: 1, 2, 3, 4, 5, 6, 7, 8    ');
disp(' ');
disp(' ');

% Potential Wells -----------------------------------------------------
switch wellType
   
% square well **********************************************************
case 1
    xMin = -0.1;         % default = -0.1 nm
    xMax = +0.1;         % default = +0.1 nm
    x1 = 0.05;            % 1/2 well width: default = 0.05 nm
    U1 = -400;            % Depth of well (eV): default = -400 eV
    
    % x1 = 1.5e-6;         % code for deuteron
    % xMin = 0;
    % xMax = +20*x1;
    % U1 = -58.5e6;
    %m = mp*mn/(mp+mn);     % reduced mass - deuteron
    
    % U1 = 0;             % code for infinite square well     
     
   x = linspace(xMin,xMax, num);
   for cn = 1 : num
   if abs(x(cn)) <= x1, U(cn) = U1; end; %if;
   end; %for
   s = sprintf('Potential Well: SQUARE');
   
% step well ************************************************************
% Asymmetrical well: comment the line %if x(cn) > x1/2, U(cn) = 0; end
    
    case 2
    xMin = -0.15;        % default = -0.15 nm
    xMax = +0.15;        % default = +0.15 nm
    x1 = 0.1;            % Total width of well: default = 0.1 nm
    x2 = 0.06;           % Width of LHS well: default = 0.06 nm;
    U1 = -400;           % Depth of LHS well; default = -400 eV;
    U2 = -250;           % Depth of RHS well (eV); default = -250 eV;
    
    x = linspace(xMin,xMax, num); 
    for cn = 1 : num
    if x(cn) >= -x1/2,      U(cn) = U1; end;  %if
    if x(cn) >= -x1/2 + x2, U(cn) = U2; end;  %if
    %if x(cn) > x1/2,        U(cn) = 0;  end; %if
        % comment above line to give an asymmetrical well
    end %for
    s = sprintf('Potential Well: STEPPED');
   
% double well ************************************************************   
case 3   
   U1 = -400;      % Depth of LHS well: default = -440 eV
   U2 = -400;      % Depth of RHS well: default = -400 eV
   U3 = -100;       % Depth of separation section: default = 100 eV 
   x1 = 0.10;      % width of LHS well: default = 0.10 nm
   x2 = 0.10;      % Width of RHS well: default = 0.10 nm
   x3 = 0.10;      % Width of separtion section: default = 0.10 nm
   xEnd = 0.05;    % parameters to define range for x-axis: default = 0.05 nm
   
   xMin = -(xEnd+x1+x3/2);
   xMax = x3/2+x2+xEnd;
   dx = (xMax-xMin)/(num-1);
   x = xMin : dx : xMax;  
        
   for cn = 1 : num
       if x(cn) >= xMin+ xEnd & x(cn) <= x1 + xMin+ xEnd, U(cn) = U1; end;
       if x(cn) >= x3/2 & x(cn) <= x2+x3/2, U(cn) = U2; end;
       if abs(x(cn)) < x3/2, U(cn) = U3; end
   end
   s = sprintf('Potential Well: DOUBLE');
   
% sloping well ***********************************************************
% Enter energies in eV and distances in nm
    case 4
    % Input parameters
    xMin = -0.1;      % default value = -0.1 nm
    xMax = +0.1;      % default value = + 0.1 nm
    U1 = -1200;        % Depth of LHS well: default = -1200 eV;
    U2 = -200;        % Depth of RHS well: default = -200 eV;
    x1 = 0.05;        % 1/2 width of well: default = 0.05 nm;
    
    x = linspace(xMin,xMax, num);
    intercept = (U1+U2)/2;
    slope = (U2-U1)/(2*x1);
    
   for cn = 1 : num
   if abs(x(cn))<= x1, U(cn) = slope * x(cn) + intercept; end;
   end %for    
   s = sprintf('Potential Well: SLOPING');
   
% parabolic ****************************************************************
case 5   
    xMin = -0.2;                  % default = -0. nm
    xMax = +0.2;                  % default = +0.2 nm
    x1 = 0.2;                     % width default = 0.2 nm;
    U1 = -400;                    % well depth default = -400 eV;
    
   x = linspace(xMin,xMax, num); 
   for cn = 1 : num
   if abs(x(cn))<=x1/2,   U(cn) = -(4*U1/(x1*x1))*x(cn)^2+U1; end;
   end %for    
   s = sprintf('Potential Well: Truncated PARABOLIC');
   
% Morse *******************************************************************
case 6   
    U1 = -1200;          % Depth of well (eV): default = -2000
    x0 = 0.08;
    S = 0.075;
   
    xMin = 0;
    xMax = 0.3;
    
    dx = (xMax-xMin)/(num-1);
    x = xMin : dx : xMax;    
    
    for cn = 1 : num
    U(cn) = U1 * (1 - (exp((x0 - x(cn))/S)-1)^2);
    end
    
    s = sprintf('Potential Well: MORSE potential');
 
   
%Parabolic fit to Morse  incomplete
% case 7   
%    disp('Change Setup values for Morse Potential');
%    for c = 1 : num
%       a1 = (U1-U2)/(R2^2);
%       a2 = (U1-U2)*2*R1/(R2^2);
%       a3 = -U1+(U1-U2)*R1^2/R2^2;
%       U(c) = a1*x(c)^2+a2*x(c)+a3;
%    	if U(c) > 400, U(c) = 400; end;   
%    end %for    
%    s = sprintf('Potential Well: PARABOLIC fit to MORSE potential');

% Lattice
case 8 
    num = 5000;
    wellNum = 12;      % number of wells
    U1 = -350;
    x1 = 0.05;
    x2 = 0.075;
    xEnd = 0.05;
    wellDepth = U1.*ones(wellNum,1);   % depth of wells
    wellWidth = x1.*ones(wellNum,1);
    wellSeparation = x2.*ones(wellNum-1,1);
    wellcenter = zeros(wellNum,1);
    xMin = 0;
    xMax = 2*xEnd + sum(wellSeparation) + 0.5*(wellWidth(1)+wellWidth(wellNum));
    x = linspace(xMin,xMax,num);
    dx = (xMax-xMin)/(num-1);
    U = zeros(num,1);
    wellcenter(1) = xMin+xEnd+wellWidth(1)/2;
    
    for cm = 2: wellNum
        wellcenter(cm) = wellcenter(cm-1) + wellSeparation(cm-1);
    end
    
    for cm = 1 : wellNum
    for cn = 1 : num 
        if abs(x(cn)-wellcenter(cm)) <= wellWidth(cm)/2; U(cn) = wellDepth(cm); end
    end
    end
     
    
    %wellDepth = U1 .* ones(numWells,1);
    %wellWidth = R1 .* ones(numWells,1);
    %defect-----------------------
    %wellDepth(5) = U2;
    %wellWidth(5) = R2;
    % defect----------------------
     %  for cn = 1 : numWells
      %     wellCentre(cn) = R1/2+ R1*(cn-1);
       %    for cm = 1: num
        %   if abs(x(cm)-wellCentre(cn)) < R2/2, U(cm) = wellDepth(cn); end
         %  end        
       %end  
   
s = sprintf('Potential Well: Lattice');

% Colvalent bonding   V shaped potentail wells   
    case 9 
    num = 1000;
    U1 = -1000;
    x1 = 0.01;
    xMin = -0.15;
    xMax = -xMin;
    x = linspace(xMin,xMax,num);
    dx = (xMax-xMin)/(num-1);
    U = zeros(num,1);
    Kc = e/(4*pi*eps0*L0);
    %Uc = 2*Kc / (x1/2+eps);
   
    for cn = 1 : num
      U(cn) =  -Kc * (1/abs(x(cn)-x1/2+eps) + 1/abs(x(cn)+x1/2+eps));
    end      
    U = U + U(end);
    
    for cn = 1 : num
       if U(cn) < U1, U(cn) = U1; end
    end    
    %W = R2/2;
        %w(1) = -R1-2*W; w(2) = w(1) + W; w(3) = w(2)+W;
        %w(4) =  R1; w(5) = w(4) + W; w(6) = w(5)+W;
        %slope(1) = U1/(w(2)-w(1)); b(1) = 0 - slope(1)*w(1);
        %slope(2) = -slope(1); b(2) = U1 - slope(2)*w(2);
        %slope(3) = slope(1); b(3) = 0 - slope(3)*w(4);
        %slope(4) = slope(2); b(4) = U1 - slope(4)*w(5);
        %Ua = 0; Ub = 0;
        %xa = x(c)-R2/2; xb = x(c)+R2/2;
   %for cn = 1 : num
        %if x(cn) > w(1) & x(cn) <= w(2), U(cn) = slope(1)*x(cn)+b(1); end
        %if x(cn) > w(2) & x(cn) <= w(3) , U(cn) = slope(2)*x(cn)+b(2); end
        %if x(cn) > w(4) & x(cn) <= w(5), U(cn) = slope(3)*x(cn)+b(3); end
        %if x(cn) > w(5) & x(cn) <= w(6) , U(cn) = slope(4)*x(cn)+b(4); end
    %    if x(cn) > w(1) & x(cn) <= w(2), U(cn) = U1; end
     %   if x(cn) > w(2) & x(cn) <= w(3) , U(cn) = U1; end
      %  if x(cn) > w(4) & x(cn) <= w(5), U(cn) = U1; end
       % if x(cn) > w(5) & x(cn) <= w(6) , U(cn) = U1; end
        
        
        %if abs(xa)<(R1/2),   Ua = (1-2*abs(xa)/R1) * U1; end;
   %if abs(xb)<(R1/2),   Ub = (1-2*abs(xb)/R1) * U1; end;
   %U(c) = Ua+Ub;
   %end  % for cn  
   s = sprintf('Potential Well: Double: V Shaped');
   
   
end;   % switch wellType

% Graphics -------------------------------------------------------------\
figure(1);
set(gcf,'Name','Potential Energy','NumberTitle','off')
plot(x,U,'LineWidth',3);
axis([xMin-eps xMax min(U)-50 max(U)+50]);

title(s);
xlabel('x   (nm)');
ylabel('energy   (eV)');
grid on

% Make potential energy matrix
dx = (x(2)-x(1));
dx2 = dx^2;


for cn =1:(num-2)
    U_matrix(cn,cn) = U(cn+1);
end;



% Matrix method used to solving time independent Schrodinger Equation
% Energies values are in eV (electron volts)
% Lengths are in nm (nanometers)
% All other units are S.I.
% You must run the m-script se_wells.m first then se_solve.m
% Solution gives the energy eigenvalues and eigenfunctions
% Outputs
         % energy eignevalues displayed in Command Window
         % Energy spectrum and Well displayed in Figure Window

tic
% Make Second Derivative Matrix ------------------------------------------
off     = ones(num-3,1);                 
SD_matrix = (-2*eye(num-2) + diag(off,1) + diag(off,-1))/dx2;

% Make KE Matrix
K_matrix = Cse * SD_matrix;            

% Make Hamiltonian Matrix
H_matrix = K_matrix + U_matrix;

% Find Eignevalues E_n and Eigenfunctions psi_N ---------------------------
[e_funct e_values] = eig(H_matrix);

% All Eigenvalues 1, 2 , ... n  where E_N < 0
flag = 0;
n = 1;
while flag == 0
    E(n) = e_values(n,n);
    if E(n) > 0, flag = 1; end; % if
    n = n + 1;
end  % while
E(n-1) = [];
n = n-2;

% Corresponding Eigenfunctions 1, 2, ... ,n: Normalizing the wavefunction
for cn = 1 : n
psi(:,cn) = [0; e_funct(:,cn); 0];
area = simpson1d((psi(:,cn) .* psi(:,cn))',xMin,xMax);
psi(:,cn) = psi(:,cn)/sqrt(area);       % normalize
prob(:,cn) = psi(:,cn) .* psi(:,cn);
if psi(5,cn) < 0, psi(:,cn) = -psi(:,cn); end;  % curve starts positive
end % for

% Display eigenvalues in Command Window -------------------------------
disp('   ');
disp('================================================================  ');
disp('  ');
fprintf('No. bound states found =  %0.0g   \n',n);
disp('   ');
disp('Quantum State / Eigenvalues  En  (eV)');

for cn = 1 : n
    fprintf('  %0.0f   ',cn);
    fprintf('   %0.5g   \n',E(cn));
end
disp('   ')
disp('   ');

% Plot energy spectrum ------------------------------------------------
xs(1) = xMin;
xs(2) = xMax;

figure(2);
set(gcf,'Units','Normalized');
set(gcf,'Position',[0.5 0.1 0.4 0.6]);
set(gcf,'Name','Energy Spectrum','NumberTitle','off')
set(gcf,'color',[1 1 1 ]);
set(gca,'fontSize',12);

plot(x,U,'b','LineWidth',2);
xlabel('position x (nm)','FontSize',12);
ylabel('energy U, E_n (eV)','FontSize',12);
h_title = title(s);
set(h_title,'FontSize',12);
hold on

cnmax = length(E);

for cn = 1 : cnmax
  ys(1) = E(cn);
  ys(2) = ys(1);
  plot(xs,ys,'r','LineWidth',2);
end %for   
axis([xMin-eps xMax min(U)-50 max(U)+50]);

% Plots first 5 wavefunctions & probability density functions
if n < 6;
    nMax = n;
else
    nMax = 5;
end;

figure(11)
clf
set(gcf,'Units','Normalized');
set(gcf,'Position',[0.05 0.1 0.4 0.6]);
set(gcf,'NumberTitle','off');
set(gcf,'Name','Eigenvectors & Prob. densities');
set(gcf,'Color',[1 1 1]);
%nMax = 8;
for cn = 1:nMax
    subplot(nMax,2,2*cn-1);
    y1 = psi(:,cn) ./ (max(psi(:,cn)-min(psi(:,cn))));
    y2 = 1 + 2 * U ./ (max(U) - min(U));
    plot(x,y1,'lineWidth',2)
    hold on
    plot(x,y2,'r','lineWidth',1)
    %plotyy(x,psi(:,cn),x,U);
    axis off
    %title('\psi cn);
    title_m = ['\psi   n = ', num2str(cn)] ;
    title(title_m,'Fontsize',10);
    
    subplot(nMax,2,2*cn);
    y1 = prob(:,cn) ./ max(prob(:,cn));
    y2 = 1 + 2 * U ./ (max(U) - min(U));
    plot(x,y1,'lineWidth',2)
    hold on
    plot(x,y2,'r','lineWidth',1)
    title_m = ['\psi^2   n = ', num2str(cn)] ;
    title(title_m,'Fontsize',10);
    axis off
end

% Graphical display for solutions of Schrodinger Eq. for a given value of n


disp('   ');
disp('   ');
qn = input('Enter Quantum Number (1, 2, 3, ...), n  =  ');

K = E(n) - U;           % kinetic energy  (eV)
EB = -E(qn);            % binding energy  (eV)

h_figure = figure(99);
clf
set(gcf,'Name','Schrodinger Equation: Bound States');
set(gcf,'NumberTitle','off');
set(gcf,'PaperType','A4');
set(gcf','Units','normalized')
set(gcf,'Position',[0.15 0.05 0.7 0.8]);
set(gcf,'Color',[1 1 1]);

axes('position',[0.1 0.6 0.35 0.32]);
[AX, H1, H2] = plotyy(x,psi(:,qn),x,U);
title_m = [s, '   n =   ', num2str(qn),'   E_n = ', num2str(E(qn))] ;
title(title_m,'Fontsize',12);
xlabel('position x (nm)')
ylabel('wave function   psi');
set(get(AX(2),'Ylabel'),'String','U  (eV)')
set(AX(1),'Xlim',[xMin-eps xMax])
set(AX(1),'YColor',[0 0 1])
set(AX(2),'Xlim',[xMin-eps xMax])
set(AX(2),'Ylim',[min(U)-50 max(K)+50])
set(AX(2),'YColor',[1 0 0])
set(H1,'color',[0 0 1],'LineWidth',3);
set(H2,'color',[1 0 0]);
%set(AX(2),'Ytick',[U1:100:0]);
%set(AX(2),'Ytick',[U1 0 -U1]);
line([xMin xMax],[-EB -EB],'LineWidth',2,'Color',[0 0 0],'Parent',AX(2))
line([xMin xMax],[0 0],'Color',[0 0 1],'Parent',AX(1))

axes('position',[0.1 0.2 0.8 0.32]);
[AX, H1, H2] = plotyy(x,prob(:,qn),x,U);
xlabel('position x (nm)')
ylabel('prob density   psi?');
set(get(AX(2),'Ylabel'),'String','U  (eV)')
set(AX(1),'Xlim',[xMin-eps xMax])
set(AX(1),'YColor',[0 0 1])
set(AX(2),'Xlim',[xMin-eps xMax])
set(AX(2),'Ylim',[min(U)-50 max(U)+50])
set(AX(2),'YColor',[1 0 0])
set(H1,'color',[0 0 1],'LineWidth',3);
set(H2,'color',[1 0 0]);
%set(AX(2),'Ytick',[U1:100:50]);
%set(AX(2),'Ytick',[U1 0 -U1]);
line([xMin xMax],[-EB -EB],'LineWidth',2,'Color',[0 0 0],'Parent',AX(2))

axes('position',[0.57 0.6 0.33 0.32]);
plot(x,U,'r','LineWidth',2);
hold on
plot(x(1:75:num),K(1:75:num),'g+');
plot(x,K,'g');
title('E_n(black)   U(red)    K(green)')  ; 
xlabel('position x (nm)');
ylabel('energies  (eV)');
line([xMin xMax],[-EB -EB],'LineWidth',2,'Color',[0 0 0]);
set(gca,'Xlim',[xMin xMax]);
set(gca,'Ylim',[min(U)-50 max(K)+50]);

% Each point represents the location of the particle after
% a measeurment is made on the system 
axes('position',[0.1 0.05 0.8 0.08]);
hold on
num1 = 10000;
axis off

for c = 1 : num1
xIndex = ceil(1+(num-2)*rand(1,1));
yIndex = rand(1,1);
pIndex = max(prob(:,n))*rand(1,1);
   if pIndex < prob(xIndex,qn);
   plot(x(xIndex),yIndex,'s','MarkerSize',2,'MarkerFaceColor','b','MarkerEdgeColor','none');
   end
end
set(gca, 'Xlim',[xMin xMax]);
set(gca, 'Ylim',[0,1]);

hold off
