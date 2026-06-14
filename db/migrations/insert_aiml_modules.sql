-- Insert AIML modules into the modules table (10 modules, 1 per CSV question)
-- Each module uses the question text as module name, and the CSV's company/role lists!

INSERT INTO modules (subdomain_id, module_name, slug, is_free, companies, job_roles) VALUES
-- Row 2: Dynamic Pricing
(3, 'Dynamic Pricing for Marketplaces', 'dynamic-pricing-marketplaces', false,
 '["Airbnb", "Ola", "Swiggy", "Amazon", "DE Shaw"]',
 '["ML Engineer", "Data Scientist", "Applied Scientist", "Quantitative Analyst"]'),

-- Row 3: Statistical Arbitrage Backtesting
(1, 'Statistical Arbitrage Backtesting', 'statistical-arbitrage-backtesting', false,
 '["Graviton Research Capital", "Squarepoint Capital", "DE Shaw", "Goldman Sachs"]',
 '["Quantitative Analyst", "Quantitative Researcher", "ML Engineer", "Data Scientist"]'),

-- Row 4: Graph Neural Networks
(1, 'Graph Neural Networks at Scale', 'graph-neural-networks-scale', false,
 '["Meta", "Apple", "Google", "Microsoft", "Amazon"]',
 '["Data Scientist", "ML Engineer", "AI/ML Engineer", "Applied Scientist"]'),

-- Row 5: Distributed Database Deadlock
(1, 'Distributed Database Deadlock Resolution', 'distributed-db-deadlock-resolution', false,
 '["Meta", "Netflix", "Amazon", "Apple", "Microsoft"]',
 '["Software Engineer (ML/AI)", "MLOps Engineer", "Backend Engineer (AI Teams)", "ML Infra Engineer"]'),

-- Row 6: Enterprise Churn Prediction
(1, 'Enterprise Churn Prediction', 'enterprise-churn-prediction', false,
 '["Siemens", "Qualcomm", "Intel", "Samsung R&D", "Apple"]',
 '["Data Scientist", "ML Engineer", "AI/ML Engineer", "Applied Scientist"]'),

-- Row 7: Imbalanced Dataset Handling
(1, 'Imbalanced Dataset Handling (0.01% Class)', 'imbalanced-dataset-handling', false,
 '["Deloitte", "EY (Ernst & Young)", "BCG (Gamma)", "Fractal Analytics", "McKinsey & Company"]',
 '["Data Scientist", "ML Engineer", "AI/ML Engineer", "Applied Scientist"]'),

-- Row 8: Real-time ML Inference
(1, 'Real-time ML Inference at Scale', 'realtime-ml-inference-scale', false,
 '["Meta", "Microsoft", "Amazon", "Netflix", "Google"]',
 '["Software Engineer (ML/AI)", "MLOps Engineer", "Backend Engineer (AI Teams)", "ML Infra Engineer"]'),

-- Row 9: Adaptive Fraud Detection
(1, 'Adaptive Fraud Detection', 'adaptive-fraud-detection', false,
 '["Razorpay", "CRED", "PayU", "Navi Technologies", "PhonePe"]',
 '["Data Scientist", "ML Engineer", "AI/ML Engineer", "Applied Scientist"]'),

-- Row 10: Predictive Maintenance
(1, 'Predictive Maintenance Optimization', 'predictive-maintenance-optimization', false,
 '["Intel", "Samsung R&D", "Texas Instruments", "Siemens", "Qualcomm"]',
 '["Data Scientist", "ML Engineer", "AI/ML Engineer", "Applied Scientist"]'),

-- Row 11: PyTorch Optimization
(1, 'PyTorch Model Inference Optimization', 'pytorch-inference-optimization', false,
 '["Amazon", "Flipkart", "Swiggy", "Zomato", "Meesho"]',
 '["Software Engineer (ML/AI)", "MLOps Engineer", "Backend Engineer (AI Teams)", "ML Infra Engineer"]');
