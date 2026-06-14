-- Insert single AIML module into the modules table (subdomain_id=3)
-- All 10 questions from the CSV are included in the module_json_path!

INSERT INTO modules (subdomain_id, module_name, slug, is_free, companies, job_roles, module_json_path) VALUES
(3, 'Advanced AIML Interview Prep', 'advanced-aiml-interview-prep', false,
 -- All unique companies from updated CSV
 '["Texas Instruments", "Qualcomm", "Intel", "NVIDIA", "Samsung R&D", "EY (Ernst & Young)", "McKinsey & Company", "EXL Services", "BCG (Gamma)", "Deloitte", "Flipkart", "Swiggy", "Zomato", "Ola", "Amazon", "Philips", "Bosch", "Siemens", "Google", "Meta", "Microsoft", "Apple", "Netflix", "Deutsche Bank", "Wells Fargo", "JP Morgan", "Goldman Sachs", "DE Shaw", "Meesho"]',
 -- All unique job roles from updated CSV
 '["Data Scientist", "ML Engineer", "AI/ML Engineer", "Applied Scientist", "Research Scientist", "Research Intern", "AI Researcher"]',
 'scripts/aiml_interview_questions.json');
