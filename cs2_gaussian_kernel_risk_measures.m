clear all
close all

%%% The file loaded here is a 7x929 matrix where each row corresponds to
%%% 929 daily returns of a different cryptocurrency, ending in February
%%% 2018

%%% For convenience, next to each crypto I report the p-values obtained
%%% from the two-sided KS test between the test set and the random numbers
%%% generated from the kernel density (in one experiment, a different one
%%% will yield similar values but different from the ones reported below).
%%% These values are meant as a guide to analyse the results on risk
%%% measures. In general, we expect worse kernel densities (i.e., those
%%% associated with lower p-values) to also perform worse in terms of risk
%%% analysis.

%%% Row 1: Bitcoin      (p = 6.14e-13)
%%% Row 2: Dash         (p = 2.17e-02)
%%% Row 3: Ethereum     (p = 6.31e-04)
%%% Row 4: Litecoin     (p = 6.32e-07)
%%% Row 5: Monero       (p = 6.41e-03)
%%% Row 6: Nem          (p = 3.63e-01)
%%% Row 7: Ripple       (p = 1.04e-03)

b = load('cryptocurrency_prices.txt'); % Loading the file

pt = 0.33; % Fraction of data to use in training set
pv = 0.33; % Fraction of data to use in validation set

alpha = 0.95; % Significance level for VaR / ES

n_rand = 5000; % Number of random numbers to be generated

var_hist = []; var_kernel = []; var_real = [];
es_hist = []; es_kernel = []; es_real = [];

for i = 1:7
    
    r = b(i,:); % Here we import the i-th cryptocurrency

    r = log(r(2:end)./r(1:end-1)); % Log-returns

    %%% Separating the data into training / validation / testing sets
    
    train_set = r(1:round(pt*length(r)));
    N_T = length(train_set);
    val_set = r(round(pt*length(r)+1:round((pt+pv)*length(r))));
    test_set = r(round((pt+pv)*length(r))+1:end);
    
    f = @(x,h,u) sum(1+erf((x-train_set)/sqrt(2*h^2)))/(2*N_T) - u;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% Maximum-likelihood analysis %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    h = logspace(-3,-1,100); % Logarithmic space for parameter h

    L = []; % Empty vector to store all log-likelihood values

    for j = 1:length(h)

        %%% Computing log-likelihood for Gaussian kernel evaluated on the
        %%% validation set (using the training set as fixed parameters

        p = gaussian_mix(val_set,train_set,h(j)); % Vector of Gaussian kernel values calculated in each point of validation set 

        aux = sum(log(p));

        L = [L; aux];

    end

    h_opt = h(find(L == max(L))); % Identifying optimal value (i.e., argmax of log-likelihood)
    
    kernel_random_numbers = []; % Vector to store all random numbers to be generated

    for n = 1:n_rand

        u = rand; % Random number drawn from uniform distribution over [0,1] 

        y = fzero(@(x) f(x,h_opt,u),0); % Numerical solution of the equation C(y)-u = 0

        kernel_random_numbers = [kernel_random_numbers; y];

    end

    %%% Computing VaR and Expected Shortfall over training + validation
    %%% sets (historical estimates)
    
    hist_set = [train_set val_set]; % Collecting training + validation sets into a single set (historical data)
    
    var_hist = [var_hist; abs(quantile(hist_set,1-alpha))]; % Historical VaR estimate
    
    var_kernel = [var_kernel; abs(quantile(kernel_random_numbers,1-alpha))];
    
    es_hist = [es_hist; abs(mean(hist_set(find(hist_set < -var_hist(end,:)))))]; % Historical ES estimate
    
    es_kernel = [es_kernel; abs(mean(kernel_random_numbers(find(kernel_random_numbers < -var_kernel(end,:)))))];
    
    %%% Realized VaR and ES (i.e., computed on test set)
    
    var_real = [var_real; abs(quantile(test_set,1-alpha))];
    
    es_real = [es_real; abs(mean(test_set(find(test_set < -var_real(end,:)))))];
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

subplot(1,2,1)
plot(1:1:7,var_hist,'ob','MarkerSize',12,'MarkerFaceColor','blue')
hold on
plot(1:1:7,var_kernel,'sm','MarkerSize',12,'MarkerFaceColor','magenta')
hold on
plot(1:1:7,var_real,'dr','MarkerSize',12,'MarkerFaceColor','red')
xlim([0 8])
ylim([0 0.15])
xticks(1:1:7)
xticklabels({'$\mathrm{Bitcoin}$','$\mathrm{Dash}$','$\mathrm{Ethereum}$',...
             '$\mathrm{Litecoin}$','$\mathrm{Monero}$','$\mathrm{Nem}$','$\mathrm{Ripple}$'})
xtickangle(45)
ylabel('$\mathrm{VaR}$','Interpreter','LaTex')
set(gca,'FontSize',20)
set(gca,'TickLabelInterpreter','LaTex')
legend({'$\mathrm{historical}$','$\mathrm{kernel}$','$\mathrm{realized}$'},'Interpreter','LaTex','Location','southeast')

subplot(1,2,2)
plot(1:1:7,es_hist,'ob','MarkerSize',12,'MarkerFaceColor','blue')
hold on
plot(1:1:7,es_kernel,'sm','MarkerSize',12,'MarkerFaceColor','magenta')
hold on
plot(1:1:7,es_real,'dr','MarkerSize',12,'MarkerFaceColor','red')
xlim([0 8])
xticks(1:1:7)
xticklabels({'$\mathrm{Bitcoin}$','$\mathrm{Dash}$','$\mathrm{Ethereum}$',...
             '$\mathrm{Litecoin}$','$\mathrm{Monero}$','$\mathrm{Nem}$','$\mathrm{Ripple}$'})
xtickangle(45)
ylabel('$\mathrm{Expected \ Shortfall}$','Interpreter','LaTex')
set(gca,'FontSize',20)
set(gca,'TickLabelInterpreter','LaTex')
