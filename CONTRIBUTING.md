## Contributing to Navigator Helpers 💻

Thanks for the interest in contributing to Navigator Helpers! We are excited to have you here.
Here's a short list of things you need to know before you start contributing:
- We are targeting Python 3.9+ for this project.
- Each change needs to be reviewed.
  - There are checks that run on each PR and they need to pass.

### Development Setup

To get started with the development environment, follow these steps:

1. Clone the repository:
   ```bash
   git clone git@github.com:gretelai/navigator-helpers.git
   ```
   
2. Create a virtual environment and activate it:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. Install the package in editable mode with `dev` extras:
   ```bash
   make pip_install_dev
   ```

### Tools

Common development tasks are available as `make` commands:
- Apply consistent formatting:
  ```bash
  make style
  ```
- Run tests:
  ```bash
  make test
  ```