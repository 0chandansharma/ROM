# ROM

# ROM: Range of Motion Assessment Tool

A modern, open-source tool for assessing joint range of motion in remote physiotherapy settings. ROM analyzes videos of movement patterns to provide objective measurements and assessment of musculoskeletal function.

## Features

- **Specialized Movement Assessments**: Targeted analysis for specific therapeutic movements
- **Clinical Measurement**: Calculates range of motion with reference to clinical norms
- **Visual Feedback**: Real-time visualization of joint angles with clinical context
- **Assessment Reports**: Generates detailed reports with clinical metrics
- **Progress Tracking**: Compares assessments over time to track rehabilitation progress

## Supported Assessments

- Shoulder flexion/abduction
- Elbow flexion/extension
- Hip flexion/extension
- Knee extension
- Ankle dorsiflexion
- Neck rotation
- Wrist flexion
- Trunk flexion

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ROM.git
cd ROM

# Install the package
pip install -e .

# For additional pose estimation capabilities:
pip install -e ".[rtmlib]"  # For RTMLib
pip install -e ".[mediapipe]"  # For MediaPipe
```

## Quick Start

```python
from ROM import core

# Process a video with default settings
core.process("path/to/your/video.mp4")

# Process with a configuration file
core.process("config.toml")
```

Or use the command-line interface:

```bash
# Basic usage
rom --video_input path/to/your/video.mp4 --assessment shoulder_flexion

# Using a configuration file
rom --config config.toml
```

## Example Configuration

```toml
[project]
video_input = "patient_exercise.mp4"
assessment = "shoulder_flexion"

[process]
show_realtime_results = true
save_vid = true
save_angles = true

[assessment]
show_normal_rom = true
```

See the `config_example.toml` file for a complete example with all available options.

## Requirements

- Python 3.7+
- OpenCV
- NumPy
- Pandas
- One of the supported pose estimation libraries:
  - RTMLib (recommended)
  - MediaPipe

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This tool is intended for educational and research purposes only. It should not replace professional medical or physiotherapy assessment. Always consult healthcare professionals for medical advice, diagnosis, or treatment.
