-- Migration to insert a targeted Advanced AI/ML & Applied Science module for the CSV questions
INSERT INTO modules (
    subdomain_id,
    module_name,
    slug,
    module_json_path,
    model_pkl_path,
    dataset_json_path,
    is_free,
    companies,
    job_roles
) VALUES (
    3,
    'Advanced AI/ML & Applied Science',
    'advanced-ai-ml-applied-science',
    'pickles/ai-mock.pkl',
    NULL,
    NULL,
    TRUE,
    '["Amazon", "Flipkart", "TCS", "Infosys", "Google", "Microsoft", "Goldman Sachs", "American Express", "Philips", "IBM", "Accenture", "NVIDIA", "Meta", "Samsung R&D", "Adobe", "Qualcomm", "Apple", "Bosch", "Glean", "Netflix", "Wipro", "Mphasis", "Zomato", "Airbnb", "Ola", "Swiggy", "DE Shaw"]',
    '["Data Scientist", "ML Engineer", "Applied Scientist", "Quantitative Engineer", "AI/ML Engineer", "Deep Learning Engineer", "Research Scientist", "Research Engineer", "AI Researcher", "ML Research Engineer", "AI/ML Research Scientist", "AI Engineer", "NLP/Research Scientist", "Software Engineer (ML Infra)", "Data Engineer", "AI Developer", "AI/ML Developer", "Quantitative Analyst"]'
)