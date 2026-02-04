
ls
vigate to your project folder
cd /path/to/your/project

# Create file with cat command
cat > README.md << 'EOF'
# 🏦 OSFI Compliance AI Assistant

AI-powered regulatory guidance for mortgage underwriting based on OSFI Guideline B-20.

## Features

- **RAG-based answers**: Retrieves relevant sections from OSFI B-20 guidelines
- **Structured responses**: OVERVIEW, KEY REQUIREMENTS, PRACTICAL APPLICATION, EXAMPLES
- **Source citations**: Every answer cites specific page numbers
- **Confidence scoring**: Indicates retrieval quality

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt