# Contributing to Valquiria Data Analysis Suite

Thank you for your interest in contributing to the Valquiria Data Analysis Suite! This project supports space analog research and physiological data analysis.

## 🚨 Important Notice

**This is a research project for scientific and educational purposes only.**

- ❌ NOT for operational military deployment
- ❌ NOT for clinical diagnosis or treatment
- ✅ Research, education, and scientific collaboration only

## 🤝 How to Contribute

### Types of Contributions Welcome

1. **Bug Reports**: Help us identify and fix issues
2. **Feature Suggestions**: Propose new analysis methods or improvements
3. **Documentation**: Improve guides, examples, and scientific explanations
4. **Code Contributions**: Implement features, fix bugs, optimize performance
5. **Testing**: Add test cases and improve test coverage
6. **Scientific Validation**: Review analysis methods and statistical approaches

### Before Contributing

1. **Read the Documentation**: Review the `docs/` folder thoroughly
2. **Check Issues**: Look for existing issues or feature requests
3. **Understand the Science**: This is a physiological data analysis platform
4. **Follow Ethics**: Respect research ethics and data privacy

## 🛠️ Development Setup

### Prerequisites

- Python 3.8+ (tested up to 3.11)
- Git
- Virtual environment (strongly recommended)

### Setup Instructions

```bash
# Clone the repository
git clone <repository-url>
cd Valquiria-Data-Analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest black flake8 mypy

# Run tests to verify setup
python -m pytest tests/
```

## 📝 Contribution Process

### 1. Fork & Branch

```bash
# Fork the repository on GitHub
# Clone your fork
git clone https://github.com/YOUR_USERNAME/Valquiria-Data-Analysis.git
cd Valquiria-Data-Analysis

# Create a feature branch
git checkout -b feature/your-feature-name
```

### 2. Make Changes

- **Code Style**: Follow PEP 8, use Black formatter
- **Documentation**: Update relevant docs for any changes
- **Tests**: Add or update tests for new features
- **Commits**: Use clear, descriptive commit messages

### 3. Test Your Changes

```bash
# Format code
black src/ tests/ examples/ scripts/

# Run linting
flake8 src/ tests/ examples/ scripts/

# Run tests
python -m pytest tests/ -v

# Test specific components
python tests/test_libraries.py
```

### 4. Submit Pull Request

1. Push your branch to your fork
2. Create a Pull Request on GitHub
3. Provide clear description of changes
4. Reference any related issues
5. Wait for review and address feedback

## 🧪 Testing Guidelines

### Test Categories

1. **Unit Tests**: Test individual functions and classes
2. **Integration Tests**: Test component interactions
3. **System Tests**: Test complete workflows
4. **Performance Tests**: Verify optimization improvements

### Writing Tests

```python
import pytest
from src.hexoskin_analyzer import HexoskinWavLoader

def test_wav_loading():
    """Test WAV file loading functionality."""
    loader = HexoskinWavLoader()
    # Your test code here
    assert loader is not None

def test_statistical_analysis():
    """Test statistical analysis methods."""
    # Test with sample data
    pass
```

### Test Data

- Use synthetic data for unit tests
- Never commit real physiological data
- Create minimal test datasets
- Document test data requirements

## 📚 Documentation Standards

### Code Documentation

```python
def calculate_hrv_metrics(rr_intervals):
    """
    Calculate HRV metrics from RR intervals.
    
    Args:
        rr_intervals (array): RR interval data in milliseconds
        
    Returns:
        dict: Dictionary containing HRV metrics
        
    Raises:
        ValueError: If input data is invalid
        
    Example:
        >>> metrics = calculate_hrv_metrics([800, 820, 810])
        >>> print(metrics['rmssd'])
    """
    pass
```

### Scientific Documentation

- Reference peer-reviewed sources
- Explain algorithm choices
- Document statistical methods
- Include validation results

## 🔬 Scientific Contributions

### Analysis Methods

- Cite relevant literature
- Validate against established methods
- Provide statistical justification
- Consider physiological relevance

### New Features

- Scientific rationale required
- Validation with known datasets
- Performance impact analysis
- Clinical interpretation guidance

## 🐛 Bug Reports

### Use GitHub Issues

Include:
- **Environment**: OS, Python version, dependencies
- **Steps to Reproduce**: Detailed instructions
- **Expected Behavior**: What should happen
- **Actual Behavior**: What actually happens
- **Screenshots**: If applicable
- **Data**: Sample data (anonymized)

### Critical Bugs

For security or data integrity issues:
1. **DO NOT** create public issues
2. Contact development team directly
3. Provide detailed technical information
4. Allow time for fix before disclosure

## 💡 Feature Requests

### Proposal Format

1. **Scientific Justification**: Why is this needed?
2. **Use Case**: How will it be used?
3. **Implementation Ideas**: Technical approach
4. **References**: Relevant literature
5. **Testing Strategy**: How to validate

### Prioritization Criteria

- Scientific value
- Research community need
- Implementation feasibility
- Maintenance burden
- Performance impact

## 📋 Code Review Process

### What We Look For

- **Correctness**: Does the code work?
- **Scientific Accuracy**: Are methods valid?
- **Performance**: Is it efficient?
- **Maintainability**: Is code clean and documented?
- **Testing**: Are there adequate tests?

### Review Timeline

- Initial response: 1-3 days
- Full review: 1-2 weeks
- Complex features: 2-4 weeks

## 🏆 Recognition

### Contributors

All contributors are acknowledged in:
- README.md contributors section
- Release notes
- Academic publications (when applicable)

### Types of Recognition

- Code contributions
- Bug reports and fixes
- Documentation improvements
- Scientific review and validation
- Testing and quality assurance

## 📞 Getting Help

### Communication Channels

- **GitHub Issues**: Technical questions and bugs
- **GitHub Discussions**: General questions and ideas
- **Documentation**: Check `docs/` folder first

### Response Times

- Bug reports: 1-3 days
- Feature requests: 1 week
- General questions: 2-5 days

## 📄 Legal & Ethics

### Code License

- All contributions subject to project license
- By contributing, you agree to license terms
- Maintain original copyright notices

### Research Ethics

- Respect data privacy
- Follow institutional guidelines  
- Acknowledge data sources
- Maintain research integrity

### Data Policy

- No real physiological data in repository
- Use synthetic or anonymized data for examples
- Follow GDPR and privacy regulations
- Respect participant confidentiality

---

## Thank You! 🚀

Your contributions help advance space medicine research and physiological data analysis. Every contribution, no matter how small, helps push the boundaries of human space exploration research.

**Together, we're building tools for the future of human space flight! 🌌** 