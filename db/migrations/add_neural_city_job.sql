-- Migration to add the Applied GenAI Intern and Full-Stack Product Engineering Intern job postings at Neural City

INSERT INTO jobs (
    corporate_user_id, title, company, work_mode, start_date, duration, stipend, apply_by, type, schedule, posted, 
    skills, description, responsibilities, additional_note, who_can_apply, other_requirements, perks, 
    company_about, opps_posted, candidates_hired
) VALUES 
(
    31, 'Applied GenAI Intern', 'Neural City', 'Remote', 'Immediately', '4 to 6 Months', '₹10,000 to ₹12,000 per month', '30 Jun 2026', 'Internship', 'Full time', 'Just now',
    '["Python", "Generative AI", "LLMs", "RAG", "Agentic Workflows", "OCR", "Vision-Language Models (VLMs)", "Geospatial Data (GIS)", "FastAPI", "React"]',
    'You''ll work at the intersection of generative AI, domain expertise, and real-world deployment challenges. Turning rough ideas into working proof-of-concepts in weeks, not months.

We are currently seeking a highly motivated Applied Gen AI Intern to join our team and contribute to developing innovative solutions that classify and rate existing city infrastructure and services parameters, such as roads, footpath, cleanliness for smart monitoring systems.

At Neural City, we build AI systems for messy, real-world governance problems not chatbot demos. You''ll work on rapid AI prototyping across urban governance, geospatial systems, infrastructure monitoring, document intelligence, and public-service workflows. This role is ideal for someone who wants to build things that interact with actual users, government systems, imperfect data, maps, images, PDFs, videos, and operational constraints.

What You''ll Learn:
- Applied AI Beyond Hype: How real AI systems are built under constraints like latency, poor data quality, governance requirements, and operational ambiguity.
- Multimodal + Agentic Systems: Hands-on exposure to OCR, RAG, VLMs, structured extraction, autonomous workflows, and AI-assisted decision systems.
- Government + CivicTech Workflows: How AI interacts with compliance, procurement, public systems, auditability, and large-scale operational realities.
- Full-Stack AI Prototyping: Frontend interfaces, backend orchestration, APIs, vector databases, cloud deployment, and rapid iteration cycles.
- Startup Execution: How to ship fast without waiting for perfect specs, perfect datasets, or perfect certainty.
- Thinking, Not Just Coding: How to break down ambiguous problems, ask better questions, and design systems instead of merely calling APIs.',
    '["Rapid AI Prototyping: Build end-to-end prototypes for government and urban use cases using LLMs, multimodal AI, RAG pipelines, agentic workflows, OCR/document intelligence, and VLMs.", "Work With Real-World Data: Handle noisy datasets including PDFs, scanned documents, maps, street imagery, GPS traces, videos, spreadsheets, and public datasets instead of benchmark-only toy problems.", "Problem Discovery & AI Research: Research domains deeply using AI tools, identify operational bottlenecks, translate workflows into AI-assisted systems, and evaluate technical feasibility quickly.", "AI Stack Selection & Integration: Experiment with emerging AI tooling including Claude API, LangChain, LlamaIndex, vector databases, open-source models, local inference setups, and multimodal pipelines.", "Geospatial + Vision Workflows: Work on AI systems involving street imagery, infrastructure analysis, GIS layers, visual scoring systems, and city-scale observations.", "Iteration & Validation: Ship prototypes fast, test with stakeholders, gather feedback, break things, improve them, and document learnings for production readiness.", "Technical Documentation: Write clean architecture notes, workflows, API documentation, deployment guides, and handoff material for future scaling."]',
    'Tentatively between ₹10,000 to ₹12,000 per month. Higher for candidates with suitable job and cultural fit.',
    '["Strong Python fundamentals: Comfortable learning new frameworks, debugging independently, and shipping quickly.", "Curious About Applied AI: You experiment beyond tutorials and are genuinely interested in practical LLM and multimodal systems.", "Comfortable With Ambiguity: Can work through unclear requirements, changing priorities, and incomplete information.", "Strong Builder Mindset: You like prototyping, testing ideas, and figuring things out independently."]',
    '["LangChain / LlamaIndex", "OCR or document AI", "GIS or geospatial data", "React / Next.js", "FastAPI", "Vision-language models", "Open-source model deployment"]',
    '["Certificate", "Flexible work hours", "Job offer (PPO possibility based on performance)"]',
    'At Neural City, our vision is to enhance the liveability of Indian cities by revolutionizing city governance through technology. Neural City (www.neuralcity.in) is a bootstrapped startup focused on creating transformative solutions in the B2G (Business to Government) domain. We aim to enhance urban planning, management, and monitoring using cutting-edge technologies.',
    1, 0
),
(
    31, 'Full-Stack Product Engineering Intern', 'Neural City', 'Remote', 'Immediately', '4 to 6 Months', '₹8,000 to ₹12,000 per month', '30 Jun 2026', 'Internship', 'Full time', 'Just now',
    '["React", "Next.js", "FastAPI", "Python", "React Native", "PostgreSQL", "PostGIS", "Redis", "Docker", "Leaflet.js", "Claude API", "LangChain", "OCR pipelines", "GIS systems", "vector databases", "telemetry systems", "asynchronous architectures"]',
    'Work on real systems designed under operational constraints including latency, unreliable data, scaling challenges, and evolving requirements.

At Neural City (www.neuralcity.in), we build real-world systems involving AI, geospatial intelligence, street-level visual data, telemetry pipelines, and operational platforms for cities and governance workflows.

This is not a "build another CRUD dashboard" internship. You''ll work on products that interact with maps, cameras, videos, GPS streams, APIs, and messy real-world datasets. We are looking for builders who enjoy complexity, ownership, rapid experimentation, and learning by shipping.

This is a hands-on role for people who want to build real systems, not just discuss them.

What You''ll Learn:
- How production-grade engineering systems are built beyond tutorials and toy projects.
- Modern AI-native engineering workflows involving coding agents, automation, retrieval systems, and multimodal AI.
- How geospatial systems, telemetry, street imagery, and operational city data become usable intelligence platforms.
- Systems thinking around scalability, reliability, asynchronous workflows, observability, and infrastructure design.
- How startups prototype, ship, iterate, and learn quickly under real operational constraints.',
    '["Build scalable full-stack applications across web, backend, mobile, GIS, and AI-assisted workflows", "Develop APIs, asynchronous services, media-processing systems, telemetry pipelines, and operational dashboards", "Work on React/Next.js frontends, FastAPI backends, React Native mobile applications, and geospatial interfaces", "Contribute to AI-assisted engineering workflows involving coding agents, OCR pipelines, multimodal systems, and intelligent automation", "Build systems involving street imagery, GPS traces, maps, infrastructure data, and operational analytics", "Participate in rapid prototyping, experimentation, debugging, performance optimization, and production-oriented engineering workflows"]',
    'Tentatively between ₹8,000 to ₹12,000 per month. Higher for candidates with suitable job and cultural fit.',
    '["Strong engineering fundamentals and ability to learn unfamiliar systems quickly", "Comfortable with ambiguity, debugging, experimentation, and ownership", "Interested in AI systems, scalable engineering, GIS, backend systems, or modern product infrastructure", "Builder mindset with curiosity to work across frontend, backend, infrastructure, and AI workflows"]',
    '["React, Next.js, FastAPI, Python, React Native", "PostgreSQL/PostGIS, Redis, Docker, Leaflet.js", "Claude API, LangChain, OCR pipelines, GIS systems", "vector databases, telemetry systems, and asynchronous architectures"]',
    '["Certificate", "Flexible work hours", "Job offer (PPO possibility based on performance)"]',
    'At Neural City, our vision is to enhance the liveability of Indian cities by revolutionizing city governance through technology. Neural City (www.neuralcity.in) is a bootstrapped startup focused on creating transformative solutions in the B2G (Business to Government) domain. We aim to enhance urban planning, management, and monitoring using cutting-edge technologies.',
    2, 0
);
