#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Assessment Reporting Module

This module provides functionality for generating assessment reports
and visualizing ROM metrics in formats suitable for clinical use.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
import os

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False
    logging.warning("ReportLab not installed. PDF reports will not be available.")


def generate_assessment_report(output_path, assessment_type, metrics, angle_data, assessment_config):
    """
    Generate a clinical assessment report
    
    Args:
        output_path: Path to save the report
        assessment_type: Type of assessment performed
        metrics: Dictionary of ROM metrics
        angle_data: DataFrame of angle measurements
        assessment_config: Assessment configuration
    
    Returns:
        Path to the generated report
    """
    # Create directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate report based on format
    report_format = output_path.suffix.lower()
    
    if report_format == '.pdf' and HAS_REPORTLAB:
        return generate_pdf_report(output_path, assessment_type, metrics, angle_data, assessment_config)
    elif report_format in ['.json', '.txt', '.csv']:
        return generate_text_report(output_path, assessment_type, metrics, angle_data, assessment_config, report_format)
    else:
        # Default to HTML if format not supported or ReportLab not installed
        if report_format == '.pdf' and not HAS_REPORTLAB:
            output_path = output_path.with_suffix('.html')
            logging.warning("ReportLab not installed. Generating HTML report instead.")
        
        return generate_html_report(output_path, assessment_type, metrics, angle_data, assessment_config)


def generate_pdf_report(output_path, assessment_type, metrics, angle_data, assessment_config):
    """
    Generate a PDF assessment report using ReportLab
    
    Args:
        output_path: Path to save the PDF report
        assessment_type: Type of assessment performed
        metrics: Dictionary of ROM metrics
        angle_data: DataFrame of angle measurements
        assessment_config: Assessment configuration
    
    Returns:
        Path to the generated PDF report
    """
    if not HAS_REPORTLAB:
        logging.error("ReportLab is required for PDF report generation.")
        return None
    
    # Create PDF document
    doc = SimpleDocTemplate(str(output_path), pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Add title
    title_style = styles["Heading1"]
    title = f"ROM Assessment Report: {assessment_type.replace('_', ' ').title()}"
    story.append(Paragraph(title, title_style))
    story.append(Spacer(1, 12))
    
    # Add date and time
    date_style = styles["Normal"]
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    story.append(Paragraph(f"Assessment Date: {date_str}", date_style))
    story.append(Spacer(1, 12))
    
    # Add description
    desc_style = styles["Normal"]
    description = assessment_config.get('description', '')
    story.append(Paragraph(f"Assessment: {description}", desc_style))
    story.append(Spacer(1, 20))
    
    # Add ROM metrics table
    story.append(Paragraph("Range of Motion Metrics:", styles["Heading2"]))
    story.append(Spacer(1, 10))
    
    # Create ROM metrics table
    metrics_table_data = [["Joint", "Side", "Min (°)", "Max (°)", "Range (°)", "Status"]]
    
    # Get normal ROM values for comparison
    normal_min = assessment_config.get('normal_min', 0)
    normal_max = assessment_config.get('normal_max', 180)
    normal_range = normal_max - normal_min
    
    # Add metrics for each joint
    for joint, data in metrics.items():
        # Check if we have data for both sides
        if isinstance(data, dict) and ('right' in data or 'left' in data):
            for side, side_data in data.items():
                # Calculate percentage of normal ROM
                achieved_range = side_data.get('range', 0)
                percent_of_normal = (achieved_range / normal_range) * 100 if normal_range > 0 else 0
                
                # Determine status
                if achieved_range >= normal_range * 0.9:
                    status = "Normal"
                elif achieved_range >= normal_range * 0.7:
                    status = "Mild limitation"
                elif achieved_range >= normal_range * 0.5:
                    status = "Moderate limitation"
                else:
                    status = "Severe limitation"
                
                metrics_table_data.append([
                    joint.title(),
                    side.title(),
                    f"{side_data.get('min', 0):.1f}",
                    f"{side_data.get('max', 0):.1f}",
                    f"{achieved_range:.1f} ({percent_of_normal:.1f}%)",
                    status
                ])
        else:
            # Single joint (e.g., neck, trunk)
            achieved_range = data.get('range', 0)
            percent_of_normal = (achieved_range / normal_range) * 100 if normal_range > 0 else 0
            
            if achieved_range >= normal_range * 0.9:
                status = "Normal"
            elif achieved_range >= normal_range * 0.7:
                status = "Mild limitation"
            elif achieved_range >= normal_range * 0.5:
                status = "Moderate limitation"
            else:
                status = "Severe limitation"
            
            metrics_table_data.append([
                joint.title(),
                "Center",
                f"{data.get('min', 0):.1f}",
                f"{data.get('max', 0):.1f}",
                f"{achieved_range:.1f} ({percent_of_normal:.1f}%)",
                status
            ])
    
    # Create table
    table = Table(metrics_table_data, colWidths=[100, 60, 60, 60, 120, 100])
    
    # Add table style
    table_style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ALIGN', (2, 1), (-2, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ])
    
    # Add alternating row colors
    for i in range(1, len(metrics_table_data)):
        if i % 2 == 0:
            table_style.add('BACKGROUND', (0, i), (-1, i), colors.lightgrey)
    
    table.setStyle(table_style)
    story.append(table)
    story.append(Spacer(1, 20))
    
    # Generate ROM time series plot
    if len(angle_data) > 0:
        plot_path = output_path.parent / f"{output_path.stem}_plot.png"
        create_rom_time_series_plot(angle_data, assessment_config, plot_path)
        
        if plot_path.exists():
            # Add plot to report
            story.append(Paragraph("ROM Time Series:", styles["Heading2"]))
            story.append(Spacer(1, 10))
            
            img = Image(str(plot_path), width=450, height=300)
            story.append(img)
            story.append(Spacer(1, 20))
    
    # Add recommendations
    story.append(Paragraph("Recommendations:", styles["Heading2"]))
    story.append(Spacer(1, 10))
    
    # Find the most limited joint
    most_limited = None
    lowest_percent = 100
    
    for joint, data in metrics.items():
        if isinstance(data, dict) and ('right' in data or 'left' in data):
            for side, side_data in data.items():
                percent = (side_data.get('range', 0) / normal_range) * 100 if normal_range > 0 else 0
                if percent < lowest_percent:
                    lowest_percent = percent
                    most_limited = f"{side} {joint}"
        else:
            percent = (data.get('range', 0) / normal_range) * 100 if normal_range > 0 else 0
            if percent < lowest_percent:
                lowest_percent = percent
                most_limited = joint
    
    # Generate recommendations based on limitation status
    if lowest_percent < 70:
        recommendations = [
            f"Focus on improving mobility in the {most_limited}.",
            "Consider the following exercises:"
        ]
        
        # Suggest exercises based on the most limited joint
        if "shoulder" in most_limited.lower():
            recommendations.extend([
                "• Pendulum exercises",
                "• Wall slides",
                "• Assisted shoulder stretches"
            ])
        elif "elbow" in most_limited.lower():
            recommendations.extend([
                "• Active-assisted elbow flexion/extension",
                "• Wrist weight exercises"
            ])
        elif "hip" in most_limited.lower():
            recommendations.extend([
                "• Hip flexor stretches",
                "• Gentle leg raises",
                "• Seated marching"
            ])
        elif "knee" in most_limited.lower():
            recommendations.extend([
                "• Heel slides",
                "• Seated knee extensions",
                "• Wall slides"
            ])
        elif "ankle" in most_limited.lower():
            recommendations.extend([
                "• Ankle pumps",
                "• Towel stretches",
                "• Calf stretches"
            ])
    else:
        recommendations = [
            "Maintenance exercises are recommended.",
            "Continue with the current exercise program."
        ]
    
    # Add recommendations to report
    for rec in recommendations:
        story.append(Paragraph(rec, styles["Normal"]))
        story.append(Spacer(1, 6))
    
    # Add footer
    story.append(Spacer(1, 40))
    footer_text = "This report was generated automatically and should be reviewed by a qualified healthcare professional."
    story.append(Paragraph(footer_text, styles["Italic"]))
    
    # Build PDF
    doc.build(story)
    
    # Clean up temporary plot file
    if 'plot_path' in locals() and plot_path.exists():
        try:
            os.remove(plot_path)
        except:
            pass
    
    return output_path


def generate_html_report(output_path, assessment_type, metrics, angle_data, assessment_config):
    """
    Generate an HTML assessment report
    
    Args:
        output_path: Path to save the HTML report
        assessment_type: Type of assessment performed
        metrics: Dictionary of ROM metrics
        angle_data: DataFrame of angle measurements
        assessment_config: Assessment configuration
    
    Returns:
        Path to the generated HTML report
    """
    # Create simple HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>ROM Assessment Report: {assessment_type}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
            h1, h2 {{ color: #2c3e50; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #2c3e50; color: white; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            .recommendations {{ background-color: #f8f9fa; padding: 15px; border-left: 4px solid #2c3e50; }}
            .footer {{ margin-top: 50px; font-style: italic; color: #7f8c8d; }}
            .plot {{ margin: 20px 0; max-width: 100%; }}
        </style>
    </head>
    <body>
        <h1>ROM Assessment Report: {assessment_type.replace('_', ' ').title()}</h1>
        <p>Assessment Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        <p>Assessment: {assessment_config.get('description', '')}</p>
        
        <h2>Range of Motion Metrics</h2>
        <table>
            <tr>
                <th>Joint</th>
                <th>Side</th>
                <th>Min (°)</th>
                <th>Max (°)</th>
                <th>Range (°)</th>
                <th>Status</th>
            </tr>
    """
    
    # Get normal ROM values for comparison
    normal_min = assessment_config.get('normal_min', 0)
    normal_max = assessment_config.get('normal_max', 180)
    normal_range = normal_max - normal_min
    
    # Add metrics for each joint
    for joint, data in metrics.items():
        # Check if we have data for both sides
        if isinstance(data, dict) and ('right' in data or 'left' in data):
            for side, side_data in data.items():
                # Calculate percentage of normal ROM
                achieved_range = side_data.get('range', 0)
                percent_of_normal = (achieved_range / normal_range) * 100 if normal_range > 0 else 0
                
                # Determine status
                if achieved_range >= normal_range * 0.9:
                    status = "Normal"
                elif achieved_range >= normal_range * 0.7:
                    status = "Mild limitation"
                elif achieved_range >= normal_range * 0.5:
                    status = "Moderate limitation"
                else:
                    status = "Severe limitation"
                
                html_content += f"""
                <tr>
                    <td>{joint.title()}</td>
                    <td>{side.title()}</td>
                    <td>{side_data.get('min', 0):.1f}</td>
                    <td>{side_data.get('max', 0):.1f}</td>
                    <td>{achieved_range:.1f} ({percent_of_normal:.1f}%)</td>
                    <td>{status}</td>
                </tr>
                """
        else:
            # Single joint (e.g., neck, trunk)
            achieved_range = data.get('range', 0)
            percent_of_normal = (achieved_range / normal_range) * 100 if normal_range > 0 else 0
            
            if achieved_range >= normal_range * 0.9:
                status = "Normal"
            elif achieved_range >= normal_range * 0.7:
                status = "Mild limitation"
            elif achieved_range >= normal_range * 0.5:
                status = "Moderate limitation"
            else:
                status = "Severe limitation"
            
            html_content += f"""
            <tr>
                <td>{joint.title()}</td>
                <td>Center</td>
                <td>{data.get('min', 0):.1f}</td>
                <td>{data.get('max', 0):.1f}</td>
                <td>{achieved_range:.1f} ({percent_of_normal:.1f}%)</td>
                <td>{status}</td>
            </tr>
            """
    
    html_content += """
        </table>
    """
    
    # Generate ROM time series plot
    if len(angle_data) > 0:
        plot_path = output_path.parent / f"{output_path.stem}_plot.png"
        create_rom_time_series_plot(angle_data, assessment_config, plot_path)
        
        if plot_path.exists():
            # Add plot to report
            plot_rel_path = plot_path.name
            html_content += f"""
            <h2>ROM Time Series</h2>
            <img src="{plot_rel_path}" alt="ROM Time Series Plot" class="plot">
            """
    
    # Find the most limited joint
    most_limited = None
    lowest_percent = 100
    
    for joint, data in metrics.items():
        if isinstance(data, dict) and ('right' in data or 'left' in data):
            for side, side_data in data.items():
                percent = (side_data.get('range', 0) / normal_range) * 100 if normal_range > 0 else 0
                if percent < lowest_percent:
                    lowest_percent = percent
                    most_limited = f"{side} {joint}"
        else:
            percent = (data.get('range', 0) / normal_range) * 100 if normal_range > 0 else 0
            if percent < lowest_percent:
                lowest_percent = percent
                most_limited = joint
    
    # Generate recommendations based on limitation status
    html_content += """
        <h2>Recommendations</h2>
        <div class="recommendations">
    """
    
    if lowest_percent < 70:
        html_content += f"""
            <p>Focus on improving mobility in the {most_limited}.</p>
            <p>Consider the following exercises:</p>
            <ul>
        """
        
        # Suggest exercises based on the most limited joint
        if "shoulder" in most_limited.lower():
            html_content += """
                <li>Pendulum exercises</li>
                <li>Wall slides</li>
                <li>Assisted shoulder stretches</li>
            """
        elif "elbow" in most_limited.lower():
            html_content += """
                <li>Active-assisted elbow flexion/extension</li>
                <li>Wrist weight exercises</li>
            """
        elif "hip" in most_limited.lower():
            html_content += """
                <li>Hip flexor stretches</li>
                <li>Gentle leg raises</li>
                <li>Seated marching</li>
            """
        elif "knee" in most_limited.lower():
            html_content += """
                <li>Heel slides</li>
                <li>Seated knee extensions</li>
                <li>Wall slides</li>
            """
        elif "ankle" in most_limited.lower():
            html_content += """
                <li>Ankle pumps</li>
                <li>Towel stretches</li>
                <li>Calf stretches</li>
            """
        
        html_content += """
            </ul>
        """
    else:
        html_content += """
            <p>Maintenance exercises are recommended.</p>
            <p>Continue with the current exercise program.</p>
        """
    
    html_content += """
        </div>
        
        <p class="footer">This report was generated automatically and should be reviewed by a qualified healthcare professional.</p>
    </body>
    </html>
    """
    
    # Write HTML to file
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    return output_path


def generate_text_report(output_path, assessment_type, metrics, angle_data, assessment_config, format_type='.txt'):
    """
    Generate a text-based assessment report (JSON, TXT, CSV)
    
    Args:
        output_path: Path to save the report
        assessment_type: Type of assessment performed
        metrics: Dictionary of ROM metrics
        angle_data: DataFrame of angle measurements
        assessment_config: Assessment configuration
        format_type: Format of the report ('.json', '.txt', '.csv')
    
    Returns:
        Path to the generated report
    """
    if format_type == '.json':
        # Create JSON report
        report_data = {
            'assessment_type': assessment_type,
            'timestamp': datetime.now().isoformat(),
            'description': assessment_config.get('description', ''),
            'metrics': metrics,
            'normal_range': {
                'min': assessment_config.get('normal_min', 0),
                'max': assessment_config.get('normal_max', 180)
            }
        }
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(i) for i in obj]
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return convert_to_serializable(obj.tolist())
            else:
                return obj
        
        report_data = convert_to_serializable(report_data)
        
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=4)
    
    elif format_type == '.csv':
        # Create CSV report
        rows = [['Assessment Type', assessment_type],
                ['Timestamp', datetime.now().isoformat()],
                ['Description', assessment_config.get('description', '')],
                ['Normal Min', assessment_config.get('normal_min', 0)],
                ['Normal Max', assessment_config.get('normal_max', 180)],
                [],
                ['Joint', 'Side', 'Min', 'Max', 'Range', 'Mean', 'Std']]
        
        # Add metrics for each joint
        for joint, data in metrics.items():
            if isinstance(data, dict) and ('right' in data or 'left' in data):
                for side, side_data in data.items():
                    rows.append([
                        joint,
                        side,
                        side_data.get('min', 0),
                        side_data.get('max', 0),
                        side_data.get('range', 0),
                        side_data.get('mean', 0),
                        side_data.get('std', 0)
                    ])
            else:
                rows.append([
                    joint,
                    'center',
                    data.get('min', 0),
                    data.get('max', 0),
                    data.get('range', 0),
                    data.get('mean', 0),
                    data.get('std', 0)
                ])
        
        # Write CSV
        with open(output_path, 'w', newline='') as f:
            import csv
            writer = csv.writer(f)
            writer.writerows(rows)
    
    else:  # Default to TXT
        # Create TXT report
        from ROM.assessment.metrics import get_assessment_summary
        summary = get_assessment_summary(metrics, assessment_config)
        
        with open(output_path, 'w') as f:
            f.write(summary)
    
    return output_path


def create_rom_time_series_plot(angle_data, assessment_config, output_path):
    """
    Create a time series plot of ROM measurements
    
    Args:
        angle_data: DataFrame of angle measurements
        assessment_config: Assessment configuration
        output_path: Path to save the plot
    
    Returns:
        Path to the saved plot
    """
    # Set up the figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get normal ROM range for reference
    normal_min = assessment_config.get('normal_min', 0)
    normal_max = assessment_config.get('normal_max', 180)
    
    # Add normal range as shaded area
    ax.axhspan(normal_min, normal_max, alpha=0.2, color='green', label='Normal Range')
    
    # Plot angle data for each joint/side
    plotted = False
    for col in angle_data.columns:
        if not angle_data[col].isna().all():
            ax.plot(angle_data.index, angle_data[col], label=col.replace('_', ' ').title(), linewidth=2)
            plotted = True
    
    if not plotted:
        # No valid data to plot
        return None
    
    # Add title and labels
    ax.set_title(f"ROM Time Series: {assessment_config['type'].replace('_', ' ').title()}")
    ax.set_xlabel('Frame')
    ax.set_ylabel('Angle (°)')
    
    # Add legend
    ax.legend(loc='upper right')
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close(fig)
    
    return output_path


def create_comparison_plot(current_metrics, previous_metrics, assessment_config, output_path):
    """
    Create a comparison plot between current and previous assessments
    
    Args:
        current_metrics: Metrics from current assessment
        previous_metrics: Metrics from previous assessment
        assessment_config: Assessment configuration
        output_path: Path to save the plot
    
    Returns:
        Path to the saved plot
    """
    if not previous_metrics:
        return None
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Prepare data for plotting
    joints = []
    current_ranges = []
    previous_ranges = []
    
    for joint, data in current_metrics.items():
        if joint in previous_metrics:
            if isinstance(data, dict) and ('right' in data or 'left' in data):
                for side in ['right', 'left']:
                    if side in data and side in previous_metrics[joint]:
                        joint_label = f"{side.title()} {joint.title()}"
                        current_range = data[side].get('range', 0)
                        previous_range = previous_metrics[joint][side].get('range', 0)
                        
                        joints.append(joint_label)
                        current_ranges.append(current_range)
                        previous_ranges.append(previous_range)
            else:
                joint_label = joint.title()
                current_range = data.get('range', 0)
                previous_range = previous_metrics[joint].get('range', 0)
                
                joints.append(joint_label)
                current_ranges.append(current_range)
                previous_ranges.append(previous_range)
    
    # Set up bar positions
    x = np.arange(len(joints))
    width = 0.35
    
    # Create bars
    ax.bar(x - width/2, previous_ranges, width, label='Previous', color='lightblue')
    ax.bar(x + width/2, current_ranges, width, label='Current', color='darkblue')
    
    # Add normal range line
    normal_range = assessment_config.get('normal_max', 180) - assessment_config.get('normal_min', 0)
    ax.axhline(y=normal_range, color='green', linestyle='--', alpha=0.7, label='Normal Range')
    
    # Add labels and title
    ax.set_xlabel('Joint')
    ax.set_ylabel('Range of Motion (°)')
    ax.set_title('ROM Comparison: Previous vs Current Assessment')
    ax.set_xticks(x)
    ax.set_xticklabels(joints, rotation=45, ha='right')
    
    # Add legend
    ax.legend()
    
    # Add grid
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close(fig)
    
    return output_path


def create_progress_chart(assessments_history, joint, side, output_path):
    """
    Create a progress chart showing ROM changes over time
    
    Args:
        assessments_history: List of dictionaries with assessment data
        joint: Joint to track
        side: Side of the body
        output_path: Path to save the chart
    
    Returns:
        Path to the saved chart
    """
    if not assessments_history or len(assessments_history) < 2:
        return None
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract dates and ROM values
    dates = []
    rom_values = []
    normal_range = None
    
    for assessment in assessments_history:
        # Extract date
        if 'timestamp' in assessment:
            try:
                date = datetime.fromisoformat(assessment['timestamp'])
                dates.append(date)
            except (ValueError, TypeError):
                # If timestamp is not valid, use current date
                dates.append(datetime.now())
        else:
            # If no timestamp, use current date
            dates.append(datetime.now())
        
        # Extract ROM value
        if 'metrics' in assessment and joint in assessment['metrics']:
            if isinstance(assessment['metrics'][joint], dict) and side in assessment['metrics'][joint]:
                rom_values.append(assessment['metrics'][joint][side].get('range', 0))
            elif not isinstance(assessment['metrics'][joint], dict):
                rom_values.append(assessment['metrics'][joint].get('range', 0))
            else:
                rom_values.append(0)
        else:
            rom_values.append(0)
        
        # Get normal range from the most recent assessment
        if normal_range is None and 'normal_range' in assessment:
            normal_range = assessment['normal_range'].get('max', 180) - assessment['normal_range'].get('min', 0)
    
    # Plot the data
    ax.plot(dates, rom_values, marker='o', linestyle='-', linewidth=2, color='blue')
    
    # Add normal range line if available
    if normal_range:
        ax.axhline(y=normal_range, color='green', linestyle='--', alpha=0.7, label='Normal Range')
    
    # Add title and labels
    joint_label = f"{side.title()} {joint.title()}" if side else joint.title()
    ax.set_title(f"{joint_label} ROM Progress Over Time")
    ax.set_xlabel('Date')
    ax.set_ylabel('Range of Motion (°)')
    
    # Format the x-axis for dates
    fig.autofmt_xdate()
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add legend
    if normal_range:
        ax.legend()
    
    # Save plot
    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close(fig)
    
    return output_path