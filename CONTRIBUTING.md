# Contributing to Paperspace AutoGluon Environment

Thank you for your interest in contributing to this project! We welcome contributions from the community.

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue with:
- A clear description of the problem
- Steps to reproduce the issue
- Expected vs actual behavior
- Your environment details (OS, Docker version, etc.)

### Suggesting Enhancements

We welcome suggestions for new features or improvements:
- Open an issue with the "enhancement" label
- Describe the feature and its benefits
- Provide examples of how it would be used

### Pull Requests

1. **Fork the repository** and create your branch from `main`
2. **Make your changes** following our guidelines below
3. **Test your changes** thoroughly
4. **Update documentation** if needed
5. **Submit a pull request** with a clear description

## Development Guidelines

### Docker Best Practices

- Keep layers minimal and combine RUN commands where possible
- Use specific package versions for reproducibility
- Clean up package caches to reduce image size
- Follow multi-stage builds if beneficial

### Code Style

- Use clear, descriptive comments
- Follow existing formatting patterns
- Test changes in a clean environment

### Testing

Before submitting a PR:

1. **Build the Docker image locally:**
   ```bash
   docker build -t test-image .
   ```

2. **Test core functionality:**
   ```bash
   docker run --rm test-image python3 -c "
   import torch
   from autogluon.timeseries import TimeSeriesPredictor
   import vectorbt as vbt
   from transformers import pipeline
   print('All imports successful!')
   "
   ```

3. **Test JupyterLab startup:**
   ```bash
   docker run -p 8888:8888 test-image start-jupyter.sh
   ```

### Package Updates

When adding or updating packages:

1. **Document the reason** for the change
2. **Check compatibility** with existing packages
3. **Update the README** if it affects user workflow
4. **Consider version pinning** for stability

### Documentation

- Update README.md for user-facing changes
- Add inline comments for complex Dockerfile sections
- Update the test notebook if adding new capabilities

## Release Process

Releases are automated through GitHub Actions:

1. **Version tags** trigger new releases
2. **Images are built** and pushed to GitHub Container Registry
3. **Security scans** are performed automatically

To create a release:
1. Update version in Dockerfile labels
2. Create and push a git tag: `git tag v1.1.0 && git push origin v1.1.0`
3. GitHub Actions will handle the rest

## Getting Help

- Open an issue for questions about contributing
- Check existing issues and PRs for similar topics
- Be patient and respectful in all interactions

## Code of Conduct

This project follows a simple code of conduct:
- Be respectful and inclusive
- Focus on constructive feedback
- Help maintain a welcoming environment for all contributors

Thank you for contributing! 🚀
