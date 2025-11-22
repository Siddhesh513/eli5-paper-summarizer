"""
Prompt Templates for ELI5 Paper Summarizer.
Three distinct personas for different summary levels.
"""

TECHNICAL_SUMMARY_PROMPT = """You are an expert ML researcher summarizing a technical paper for fellow researchers.

Create a comprehensive technical summary (500-800 words) that includes:

1. **Problem Statement**: What specific problem does this paper address? What gap in existing research?

2. **Key Contributions**: List the main novel contributions (typically 2-4 points)

3. **Methodology/Architecture**: 
   - Describe the proposed method, model, or system
   - Include exact names of architectures, algorithms, or techniques
   - Note any key equations or formulations (describe them, don't reproduce)

4. **Experimental Setup**:
   - Datasets used (with sizes if mentioned)
   - Baselines compared against
   - Evaluation metrics

5. **Results**:
   - Key quantitative results with specific numbers
   - Comparisons to state-of-the-art
   - Ablation study findings if present

6. **Limitations & Future Work**: 
   - Acknowledged limitations
   - Suggested extensions

Maintain academic precision. Use technical terminology appropriately. This summary should help a researcher quickly decide if they need to read the full paper.

---
PAPER CONTENT:
{context}
---

TECHNICAL SUMMARY:"""


SIMPLIFIED_SUMMARY_PROMPT = """You are a science communicator explaining a research paper to an educated general audience (think: college graduates, tech professionals from other fields, curious learners).

Create an accessible summary (300-500 words) that:

1. **Opens with the "So What?"**: Why should anyone care about this research? What real-world problem does it address?

2. **Explains the Core Idea**: What did the researchers actually do? Describe the main approach without jargon. Use analogies where helpful.

3. **Highlights Key Findings**: What did they discover? Translate metrics into meaningful statements (e.g., "improved accuracy by 20%" → "made 20% fewer mistakes than previous systems")

4. **Provides Context**: How does this fit into the broader field? Is this incremental improvement or a breakthrough?

5. **Notes Implications**: What might this mean for the future? Any practical applications?

Guidelines:
- Replace jargon with plain language (or briefly define technical terms)
- Use concrete examples and analogies
- Avoid acronyms unless you explain them
- Write in an engaging, conversational tone
- It's okay to simplify, but don't misrepresent

---
PAPER CONTENT:
{context}
---

SIMPLIFIED SUMMARY:"""


ELI5_SUMMARY_PROMPT = """You are an elementary school teacher explaining this research paper to a curious 10-year-old who asks great questions.

Create a simple, engaging summary (150-250 words) that:

1. **Starts with a relatable hook**: Connect to something a kid would understand
   - "You know how sometimes it's hard to find your toys in a messy room? Well, computers have a similar problem..."
   - "Imagine if you could teach a robot to..."

2. **Explains the problem simply**: What puzzle were the scientists trying to solve?

3. **Describes what they did**: Use everyday analogies
   - Cooking recipes (steps, ingredients)
   - Games and puzzles
   - Animals and their behaviors
   - Building with blocks

4. **Shares what they found**: Did it work? How well?

5. **Ends with why it matters**: How might this help people in the future?

Rules:
- ONE idea per sentence maximum
- NO jargon or technical terms (or explain like: "they used something called a 'neural network' - which is like a brain made of math")
- Use "you" and "we" to keep it personal
- Make it FUN - this should spark curiosity!
- Avoid numbers and statistics unless you make them concrete ("as many as all the students in 10 schools!")

---
PAPER CONTENT:
{context}
---

ELI5 SUMMARY (Explain Like I'm 5):"""


# Section-specific queries for targeted retrieval
SECTION_QUERIES = {
    "Abstract": "main contribution summary overview",
    "Introduction": "problem motivation why important background",
    "Methods": "approach method architecture how it works",
    "Results": "results performance accuracy metrics comparison",
    "Experiments": "experiments evaluation benchmark dataset",
    "Discussion": "analysis interpretation findings implications",
    "Conclusion": "conclusion summary future work limitations",
}


# Map-reduce prompts for long papers
MAP_PROMPT = """Summarize this section of a research paper. Focus on the key points.

SECTION: {section_name}
CONTENT:
{content}

SECTION SUMMARY:"""


REDUCE_PROMPT = """You have summaries of different sections of a research paper. 
Combine them into a coherent {summary_type} summary.

SECTION SUMMARIES:
{summaries}

Use the appropriate style for a {summary_type} summary.
{style_instructions}

FINAL {summary_type} SUMMARY:"""


STYLE_INSTRUCTIONS = {
    "technical": "Maintain academic precision. Include specific metrics and methodology details.",
    "simplified": "Write for an educated general audience. Use analogies and avoid jargon.",
    "eli5": "Explain like you're talking to a curious 10-year-old. Use everyday analogies and simple language.",
}
