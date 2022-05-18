% Testing Fourier Series

clear; close all; clc


%% Parameters
L = pi;         % 2-pi periodic function
N = 80;

% Genrate equispaced (periodic) grid in space
ngrid = 500;
h = 2*L/ngrid;      % Grid spacing
x = h*(0:ngrid-1).';


%% Generate test function
% +1/1 function (Unit Step function)
%         1,    0<=x<=L
% f(x) = -1,    L<x<2L
%

% Triangle (Sawtooth function)
%         x,       0<=x<=pi
% f(x) = x-2pi,    pi<x<2pi
%


f = zeros(ngrid, 1);

% Step function
% f = (x<=L) - (x>L);

% Sawtooth function
f = x.*(x<=L) + (x-2*pi).*(x>L);


% for idx = 1:ngrid
%     if(x(idx)<=L)
%         f(idx) = 1;
%     else
%         f(idx) = -1;
%     end
% end


% Plot test function
figure; 
plot(x, f, 'linewidth', 2); 
xlabel x
ylabel f(x)
title 'Testing Fourier Approximations'
grid on


%% Generate Fourier Series coeffcients
% Analytic expression for unit step function coefficients
% a0 = 0;
% a_n = -sin(2*n*pi)/(n*pi)
% b_n = 2(1-cos(n*pi))/(n*pi);


% Analytic expression for sawtooth function coefficients
% a0 = 0;
% a_n = 0;
% b_n = -2*cos(n*pi)/n

% Index for Fourier coefficients
n = (1:N).';

% % Unit step coefficients
% a0 = 0; 
% a_n = -sin(2*n*pi)./(n*pi);
% b_n = 2*(1-cos(n*pi))./(n*pi);

% Sawtooth coefficients
a0 = 0; 
a_n = zeros(N,1);
b_n = -2*cos(n*pi)./n;




%% Compute Fourier series approximation

f_aprx = zeros(ngrid, 1); 

for idx = 1:ngrid
    f_aprx(idx) = a0 + a_n.'*cos(n*x(idx)) + ...
                  b_n.'*sin(n*x(idx));
end

% Plot
hold on
plot(x, f_aprx, '-', 'linewidth', 2);
legend( 'True', 'Approximation' )


%% Error plot

figure; 
semilogy(x, abs(f - f_aprx), 'LineWidth', 2);
xlabel x
ylabel 'Log Abs. Error'
title 'Error in Approximation'
grid



%% Error Convergence plot

% Plot maximum errors for different N values
Nvals = [20; 40; 80; 160; 500; 1000];

% Store errors here
errors = zeros(ngrid, length(Nvals));


figure;

for iN = 1:length(Nvals)
    
    
%% Parameters
L = pi;         % 2-pi periodic function
N = Nvals(iN);

% Genrate equispaced (periodic) grid in space
ngrid = 500;
h = 2*L/ngrid;      % Grid spacing
x = h*(0:ngrid-1).';


%% Generate test function
% +1/1 function
%         1,    0<=x<=L
% f(x) = -1,    L<x<2L
%

f = zeros(ngrid, 1);

% f_alt = (x<=L) - (x>L);

for idx = 1:ngrid
    if(x(idx)<=L)
        f(idx) = 1;
    else
        f(idx) = -1;
    end
end


%% Generate Fourier Series coeffcients
% Analytic expression for coefficients
% a0 = 0;
% a_n = -sin(2*n*pi)/(n*pi)
% b_n = -cos(2*n*pi)/(n*pi);

% Index for Fourier coefficients
n = (1:N).';
a0 = 0; 
a_n = -sin(2*n*pi)./(n*pi);
b_n = 2*(1-cos(n*pi))./(n*pi);


%% Compute Fourier series approximation

f_aprx = zeros(ngrid, 1); 

for idx = 1:ngrid
    f_aprx(idx) = a0 + a_n.'*cos(n*x(idx)) + ...
                  b_n.'*sin(n*x(idx));
end
    

% Record errors
errors(:,iN) = abs(f - f_aprx);


% Plot errors at each N
semilogy(x, abs(f - f_aprx), 'LineWidth', 2); hold on
xlabel x
ylabel 'Log Abs. Error'
title 'Error in Approximation'
grid on


end





