%%0. Important constants
nGenesUse = 600; % how many genes to select
K = 3 ; %use K-parameters
n_parallel = 6;
err2weight = 1.5;
%% 1. Read the table and split in parts
tic;
datfile = 'BRCA_U133A.csv';
tbl = readtable(datfile, 'ReadRowNames', true);
n_cases =  size(tbl, 1);
gene_data= tbl{:,1:end-2};
n_genes = size(gene_data, 2);
labels = tbl{:,end-1};
set_data = tbl{:,end};
set_data_names = unique(set_data);
n_set_data_names = size(set_data_names,1);
fprintf('Read data from %s\n', datfile);
set_data_counts = zeros(1, n_set_data_names);
for i = 1:n_cases
    for j = 1:n_set_data_names
        if strcmp(set_data{i}, set_data_names{j})
            set_data_counts(j)=set_data_counts(j)+1;
        end
    end
end
            

for i =1:size(set_data_names,1)
    fprintf('%s: %d cases\n', set_data_names{i}, set_data_counts(i));
end
toc;

%% 2. Select train and test sets
%tic;
train_set = false(1, n_cases);
test_set = false(1, n_cases);
for i = 1:n_cases
    %if strcmp(set_data{i}, 'GSE3494') || strcmp(set_data{i},  'GSE6532')
    %if strcmp(set_data{i}, set_data_names{3}) ||strcmp(set_data{i}, set_data_names{4}) || strcmp(set_data{i},  set_data_names{5})
    if strcmp(set_data{i}, set_data_names{4}) || strcmp(set_data{i},  set_data_names{5})
        train_set(i) = true;
    else
        test_set(i) = true;
    end
end
gene_data_train = gene_data(train_set, :);
labels_train   = labels(train_set);
gene_data_test = gene_data(test_set, :);
labels_test_gt = labels(test_set);

gene_data_i = cell(1,n_set_data_names );
labels_i  = cell(1,n_set_data_names );
for i = 1:n_set_data_names
    gene_data_i{i} = gene_data(strcmp(set_data, set_data_names{i}), :);
    labels_i{i} = labels(strcmp(set_data, set_data_names{i}), :);
end
fprintf('Selected train(%d) and test(%d) datasets\n', sum(train_set), sum(test_set));


%% 3. Compute all correllations on train dataset
tic;
CorrCoeff = zeros(1, n_genes);
for i = 1:n_genes
    CorrCoeff(i) = corr(gene_data_train(:,i), labels_train, 'Type','Spearman');
    %CorrCoeff(i) = corr(gene_data(:,i), labels, 'Type','Spearman'); % test on all!
end
figure; hist(CorrCoeff, 50);
fprintf('Computed correllations\n');
toc;
%% 4. Select which genes to use - find the most correllated

[~, corr_order] = sort(abs(CorrCoeff), 'descend');
posAll=corr_order(1:nGenesUse);
cordat = gene_data(:, posAll);
fprintf('Selected %d genes out of %d\n', numel(posAll), n_genes);
%% 5. Test all the combinations (the longest part)
tic;

maxAUC = inf(1, n_parallel);
posBest = cell(1, n_parallel);
n_attrib = numel(posAll);

Positions = nchoosek(1:n_attrib, K);
%labels = meddat(:, end);
num_pos = size(Positions, 1);

pos_step = floor(num_pos/50/n_parallel);

ParetoCurve = cell(1, n_parallel);
parfor i = 1:n_parallel
%for i = 1:n_parallel
    %for posK = Positions'
    tic;
    iter = 1;
    pos = 0;

    maxAUCThread = -inf;
    posBestThread = [];
    ParetoCurveThread=[];
    for j = i:n_parallel:num_pos
        pos = pos + 1;
        if mod(pos, pos_step) == 0
            fprintf('Thread %d %g%% done\n', i, round(j/num_pos*100));
            if i==1, toc; end
        end
        
        posK1 = Positions(j,:);%posK';
        train_data = gene_data_train(:, posK1);

        SVMModel = fitcsvm(train_data, labels_train,...
            'Standardize', true, 'Cost',[0, 1;  err2weight, 0]) ;
        Sensitivity_s = zeros(1, n_set_data_names);
        Specificity_s = zeros(1, n_set_data_names);
        for s = 1:n_set_data_names
            gene_data_tmp = gene_data_i{s};
            labels_tmp = labels_i{s};
            [label_test_predict, scores] = predict(SVMModel, gene_data_tmp(:, posK1));
            %[X_ROC, Y_ROC, T, AUC] = perfcurve(labels_test_gt, scores(:,2), 1);
            labels_test_gt_s = labels_i{s};
            TP = dot(labels_test_gt_s, label_test_predict);
            FN = dot(labels_test_gt_s, 1-label_test_predict);
            FP = dot(1-labels_test_gt_s, label_test_predict);
            TN = dot(1-labels_test_gt_s, 1-label_test_predict);
            Sensitivity_s(s) = TP / (TP + FN);
            Specificity_s(s) = TN / (TN + FP);
        end
        Sensitivity = min(Sensitivity_s);
        Specificity = min(Specificity_s);
        
        found = false;
        for k = 1:size(ParetoCurveThread, 1)
            Sensitivity_j = ParetoCurveThread(k, 1);
            Specificity_j = ParetoCurveThread(k, 2); 
            if Sensitivity_j >= Sensitivity && Specificity_j >= Specificity
                found = true;
                break;
            end
            if Sensitivity_j <= Sensitivity && Specificity_j <= Specificity
                ParetoCurveThread(k, 1) = Sensitivity;
                ParetoCurveThread(k, 2) = Specificity; 
                ParetoCurveThread(k, 3:5) = posK1;
                found = true;
                break;
            end
        end
        if ~found
            ParetoCurveThread = [ParetoCurveThread ;  Sensitivity, Specificity, posK1];
        end
    end
    ParetoCurve{i} = ParetoCurveThread;
    maxAUC(i) = maxAUCThread;
    posBest{i} = posBestThread; 
end
toc;
%% 6. Construct Pareto Curve
% 6.1. Minimum curve
ParetoCurveMerged = [];
for i = 1:n_parallel
    ParetoCurveMerged = [ParetoCurveMerged ;ParetoCurve{i}];
end
ParetoCurveMerged  = sortrows(ParetoCurveMerged);

ispareto = true(1, size(ParetoCurveMerged,1));
for i = 1:size(ParetoCurveMerged,1)
    for j = 1:size(ParetoCurveMerged,1)
        if i == j, continue;end
        if all(ParetoCurveMerged(i,1:2) >= ParetoCurveMerged(j,1:2)) && ...
                any(ParetoCurveMerged(i,1:2) > ParetoCurveMerged(j,1:2))
            ispareto(j)=false;
        end
    end
end
Sens = ParetoCurveMerged(ispareto, 1);
Spec = ParetoCurveMerged(ispareto, 2);
Triples = ParetoCurveMerged(ispareto, 3:5);
nTriples =  size(Triples,1);
% 6.2. separate Pareto curves for each dataset
ParetoSeparate = zeros(n_set_data_names, nTriples, 2);

  
for i = 1:nTriples    
    posK_i = Triples(i,:);
    train_data = gene_data_train(:, posK_i);

    SVMModel = fitcsvm(train_data, labels_train,...
        'Standardize', true, 'Cost',[0, 1;  err2weight, 0]) ;
    %Sensitivity_i = zeros(1, n_set_data_names);
    %Specificity_i = zeros(1, n_set_data_names);
    for s = 1:n_set_data_names
        gene_data_tmp = gene_data_i{s};
        labels_tmp = labels_i{s};
        [label_test_predict, scores] = predict(SVMModel, gene_data_tmp(:, posK_i));
        labels_test_gt_s = labels_i{s};
        TP = dot(labels_test_gt_s, label_test_predict);
        FN = dot(labels_test_gt_s, 1-label_test_predict);
        FP = dot(1-labels_test_gt_s, label_test_predict);
        TN = dot(1-labels_test_gt_s, 1-label_test_predict);
        %Sensitivity_i(s) = TP / (TP + FN);
        ParetoSeparate(s,i,1) = TP / (TP + FN);
        %Specificity_i(s) = TN / (TN + FP);
        ParetoSeparate(s,i,2) = TN / (TN + FP);
    end

    % for each of nTriples compute sensitivity and specificity
    %Sens_i = zeros(1, nTriples);
    %Spec_i = zeros(1, nTriples);
end
fprintf('Constructed Pareto curves\n');

            
%% 7. Draw Pareto Curve
figure;
plot(Sens, Spec, '-x');
hold on;
plot([0 1], [0 1],'--');

[~, posM] = min(abs(Spec-Sens));
plot(Sens(posM), Spec(posM),  'mo');

for s = 1:n_set_data_names
    plot(ParetoSeparate(s,:,1), ParetoSeparate(s,:,2));
end
    

BestTriple = posAll(Triples(posM, :));
genes_names3 =[ sprintf('%s ', tbl.Properties.VariableNames{BestTriple}), sprintf(' %.3g%%', 100*[Sens(posM), Spec(posM)]) ];
ylabel('Sensitivity');
xlabel('Specificity');
leg_captions = [{'ROC(Pareto)', 'x=y', genes_names3}, set_data_names'];
%legend('ROC(Pareto)', 'x=y', genes_names3, 'Location', 'SouthEast');
legend(leg_captions, 'Location', 'SouthEast');
grid on;
fprintf('Plot Pareto curve\n');
