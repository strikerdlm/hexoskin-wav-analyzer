"""
HRV Reference Ranges Citation Display Module

This module provides a comprehensive display window for showing HRV reference
ranges with their scientific citations, allowing users to understand the
evidence base behind the normal values used in the analysis.

Author: AI Assistant
Date: 2025-01-14
Integration: Enhanced HRV Analysis System
"""

import tkinter as tk
from tkinter import ttk, scrolledtext
from typing import Dict, Any, Optional
import logging
import webbrowser

# Import HRV reference ranges
try:
    from visualization.hrv_reference_ranges import hrv_reference_ranges
except ImportError:
    try:
        from enhanced_hrv_analysis.visualization.hrv_reference_ranges import hrv_reference_ranges
    except ImportError:
        hrv_reference_ranges = None

logger = logging.getLogger(__name__)


class HRVCitationDisplayWindow:
    """Window for displaying HRV reference ranges with scientific citations."""
    
    def __init__(self, parent=None):
        """
        Initialize the citation display window.
        
        Args:
            parent: Parent window (optional)
        """
        self.parent = parent
        self.window = None
        self.setup_window()
    
    def setup_window(self):
        """Setup the main window and interface."""
        self.window = tk.Toplevel(self.parent) if self.parent else tk.Tk()
        self.window.title("HRV Reference Ranges - Scientific Citations")
        self.window.geometry("1000x700")
        
        # Make window resizable
        self.window.rowconfigure(0, weight=1)
        self.window.columnconfigure(0, weight=1)
        
        # Create main frame
        main_frame = ttk.Frame(self.window, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        main_frame.rowconfigure(1, weight=1)
        main_frame.columnconfigure(0, weight=1)
        
        # Title
        title_label = ttk.Label(
            main_frame, 
            text="Heart Rate Variability Reference Ranges",
            font=('Arial', 16, 'bold')
        )
        title_label.grid(row=0, column=0, pady=(0, 10), sticky=tk.W)
        
        # Subtitle
        subtitle_text = (
            "Normal reference values for healthy females aged 24-45 years\n"
            "Based on peer-reviewed scientific literature"
        )
        subtitle_label = ttk.Label(
            main_frame, 
            text=subtitle_text,
            font=('Arial', 11),
            foreground='gray'
        )
        subtitle_label.grid(row=0, column=1, pady=(0, 10), sticky=tk.E)
        
        # Create notebook for different domains
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create tabs
        self.create_time_domain_tab()
        self.create_frequency_domain_tab()
        self.create_nonlinear_domain_tab()
        self.create_clinical_thresholds_tab()
        self.create_citations_tab()
        
        # Buttons frame
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.grid(row=2, column=0, columnspan=2, pady=(10, 0), sticky=tk.E)
        
        # Export button
        ttk.Button(
            buttons_frame,
            text="Export Citations",
            command=self.export_citations
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        # Close button
        ttk.Button(
            buttons_frame,
            text="Close",
            command=self.close_window
        ).pack(side=tk.LEFT)
        
        logger.info("HRV citation display window initialized")
    
    def create_time_domain_tab(self):
        """Create the time domain metrics tab."""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Time Domain")
        
        # Create scrollable text widget
        text_widget = scrolledtext.ScrolledText(
            frame, 
            wrap=tk.WORD, 
            font=('Consolas', 10)
        )
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Add content
        if hrv_reference_ranges:
            content = self._generate_time_domain_content()
        else:
            content = "Reference ranges module not available."
        
        text_widget.insert(tk.END, content)
        text_widget.config(state=tk.DISABLED)
    
    def create_frequency_domain_tab(self):
        """Create the frequency domain metrics tab."""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Frequency Domain")
        
        # Create scrollable text widget
        text_widget = scrolledtext.ScrolledText(
            frame, 
            wrap=tk.WORD, 
            font=('Consolas', 10)
        )
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Add content
        if hrv_reference_ranges:
            content = self._generate_frequency_domain_content()
        else:
            content = "Reference ranges module not available."
        
        text_widget.insert(tk.END, content)
        text_widget.config(state=tk.DISABLED)
    
    def create_nonlinear_domain_tab(self):
        """Create the nonlinear metrics tab."""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Nonlinear Analysis")
        
        # Create scrollable text widget
        text_widget = scrolledtext.ScrolledText(
            frame, 
            wrap=tk.WORD, 
            font=('Consolas', 10)
        )
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Add content
        if hrv_reference_ranges:
            content = self._generate_nonlinear_domain_content()
        else:
            content = "Reference ranges module not available."
        
        text_widget.insert(tk.END, content)
        text_widget.config(state=tk.DISABLED)
    
    def create_clinical_thresholds_tab(self):
        """Create the reference ranges interpretation tab."""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Interpretation Guide")
        
        # Create scrollable text widget
        text_widget = scrolledtext.ScrolledText(
            frame, 
            wrap=tk.WORD, 
            font=('Consolas', 10)
        )
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Add content
        if hrv_reference_ranges:
            content = self._generate_clinical_thresholds_content()
        else:
            content = "Reference ranges module not available."
        
        text_widget.insert(tk.END, content)
        text_widget.config(state=tk.DISABLED)
    
    def create_citations_tab(self):
        """Create the detailed citations tab."""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Scientific Citations")
        
        # Create scrollable text widget
        text_widget = scrolledtext.ScrolledText(
            frame, 
            wrap=tk.WORD, 
            font=('Consolas', 10)
        )
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Add content
        if hrv_reference_ranges:
            content = self._generate_citations_content()
        else:
            content = "Reference ranges module not available."
        
        text_widget.insert(tk.END, content)
        text_widget.config(state=tk.DISABLED)
        
        # Make links clickable (simplified version)
        text_widget.bind("<Button-1>", self._handle_link_click)
    
    def _generate_time_domain_content(self) -> str:
        """Generate content for time domain tab."""
        content = "TIME DOMAIN HRV METRICS - REFERENCE RANGES\n"
        content += "=" * 50 + "\n\n"
        
        time_metrics = ['sdnn', 'sdnn_24h', 'rmssd', 'rmssd_24h', 'pnn50', 'pnn50_24h']
        
        for metric_key in time_metrics:
            ref_range = hrv_reference_ranges.get_range(metric_key)
            if ref_range:
                content += f"{ref_range.metric_name} ({ref_range.unit})\n"
                content += "-" * 30 + "\n"
                content += f"Domain: {ref_range.domain}\n"
                content += f"Recording: {ref_range.recording_type.value}\n"
                content += f"Population: {ref_range.population}\n"
                content += f"Sample Size: {ref_range.sample_size or 'Not specified'}\n\n"
                
                content += "Reference Ranges:\n"
                if ref_range.percentile_5:
                    content += f"  5th percentile: {ref_range.percentile_5:.1f} {ref_range.unit}\n"
                if ref_range.percentile_25:
                    content += f"  25th percentile: {ref_range.percentile_25:.1f} {ref_range.unit}\n"
                if ref_range.percentile_50:
                    content += f"  Median (50th): {ref_range.percentile_50:.1f} {ref_range.unit}\n"
                if ref_range.percentile_75:
                    content += f"  75th percentile: {ref_range.percentile_75:.1f} {ref_range.unit}\n"
                if ref_range.percentile_95:
                    content += f"  95th percentile: {ref_range.percentile_95:.1f} {ref_range.unit}\n"
                
                if ref_range.mean and ref_range.std:
                    content += f"  Mean ± SD: {ref_range.mean:.1f} ± {ref_range.std:.1f} {ref_range.unit}\n"
                
                content += f"\nNormal Range: {ref_range.percentile_25:.1f} - {ref_range.percentile_75:.1f} {ref_range.unit}\n"
                content += f"Notes: {ref_range.notes}\n"
                content += f"Citation: {ref_range.doi_pmid}\n\n"
                content += "─" * 60 + "\n\n"
        
        return content
    
    def _generate_frequency_domain_content(self) -> str:
        """Generate content for frequency domain tab."""
        content = "FREQUENCY DOMAIN HRV METRICS - REFERENCE RANGES\n"
        content += "=" * 52 + "\n\n"
        
        freq_metrics = ['hf_power', 'lf_power', 'vlf_power', 'total_power', 'lf_hf_ratio']
        
        for metric_key in freq_metrics:
            ref_range = hrv_reference_ranges.get_range(metric_key)
            if ref_range:
                content += f"{ref_range.metric_name} ({ref_range.unit})\n"
                content += "-" * 30 + "\n"
                content += f"Domain: {ref_range.domain}\n"
                content += f"Recording: {ref_range.recording_type.value}\n"
                content += f"Population: {ref_range.population}\n"
                content += f"Sample Size: {ref_range.sample_size or 'Not specified'}\n\n"
                
                content += "Reference Ranges:\n"
                if ref_range.percentile_5:
                    content += f"  5th percentile: {ref_range.percentile_5:.0f} {ref_range.unit}\n"
                if ref_range.percentile_25:
                    content += f"  25th percentile: {ref_range.percentile_25:.0f} {ref_range.unit}\n"
                if ref_range.percentile_50:
                    content += f"  Median (50th): {ref_range.percentile_50:.0f} {ref_range.unit}\n"
                if ref_range.percentile_75:
                    content += f"  75th percentile: {ref_range.percentile_75:.0f} {ref_range.unit}\n"
                if ref_range.percentile_95:
                    content += f"  95th percentile: {ref_range.percentile_95:.0f} {ref_range.unit}\n"
                
                if ref_range.mean and ref_range.std:
                    content += f"  Mean ± SD: {ref_range.mean:.0f} ± {ref_range.std:.0f} {ref_range.unit}\n"
                
                content += f"\nNormal Range: {ref_range.percentile_25:.0f} - {ref_range.percentile_75:.0f} {ref_range.unit}\n"
                content += f"Notes: {ref_range.notes}\n"
                content += f"Citation: {ref_range.doi_pmid}\n\n"
                content += "─" * 60 + "\n\n"
        
        return content
    
    def _generate_nonlinear_domain_content(self) -> str:
        """Generate content for nonlinear domain tab."""
        content = "NONLINEAR HRV METRICS - REFERENCE RANGES\n"
        content += "=" * 42 + "\n\n"
        
        nonlinear_metrics = ['sd1', 'sd2', 'sd1_sd2_ratio', 'sample_entropy', 'dfa_alpha1']
        
        for metric_key in nonlinear_metrics:
            ref_range = hrv_reference_ranges.get_range(metric_key)
            if ref_range:
                content += f"{ref_range.metric_name} ({ref_range.unit})\n"
                content += "-" * 30 + "\n"
                content += f"Domain: {ref_range.domain}\n"
                content += f"Recording: {ref_range.recording_type.value}\n"
                content += f"Population: {ref_range.population}\n"
                content += f"Sample Size: {ref_range.sample_size or 'Not specified'}\n\n"
                
                content += "Reference Ranges:\n"
                if ref_range.percentile_5:
                    content += f"  5th percentile: {ref_range.percentile_5:.2f} {ref_range.unit}\n"
                if ref_range.percentile_25:
                    content += f"  25th percentile: {ref_range.percentile_25:.2f} {ref_range.unit}\n"
                if ref_range.percentile_50:
                    content += f"  Median (50th): {ref_range.percentile_50:.2f} {ref_range.unit}\n"
                if ref_range.percentile_75:
                    content += f"  75th percentile: {ref_range.percentile_75:.2f} {ref_range.unit}\n"
                if ref_range.percentile_95:
                    content += f"  95th percentile: {ref_range.percentile_95:.2f} {ref_range.unit}\n"
                
                if ref_range.mean and ref_range.std:
                    content += f"  Mean ± SD: {ref_range.mean:.2f} ± {ref_range.std:.2f} {ref_range.unit}\n"
                
                content += f"\nNormal Range: {ref_range.percentile_25:.2f} - {ref_range.percentile_75:.2f} {ref_range.unit}\n"
                content += f"Notes: {ref_range.notes}\n"
                content += f"Citation: {ref_range.doi_pmid}\n\n"
                content += "─" * 60 + "\n\n"
        
        return content
    
    def _generate_clinical_thresholds_content(self) -> str:
        """Generate content for clinical thresholds tab."""
        content = "REFERENCE RANGES INTERPRETATION\n"
        content += "=" * 35 + "\n\n"
        content += "This section explains how to interpret HRV values relative to\n"
        content += "the reference ranges from healthy populations.\n\n"
        
        content += "PERCENTILE INTERPRETATION:\n"
        content += "-" * 28 + "\n"
        content += "• 95th percentile and above: Very high values\n"
        content += "• 75th-95th percentile: Above average\n"
        content += "• 25th-75th percentile: Normal range (optimal)\n"
        content += "• 5th-25th percentile: Below average\n"
        content += "• Below 5th percentile: Very low values\n\n"
        
        content += "CLINICAL SIGNIFICANCE:\n"
        content += "-" * 22 + "\n"
        content += "• Normal range (25th-75th percentile): Good autonomic function\n"
        content += "• Above normal: May indicate excellent cardiovascular fitness\n"
        content += "• Below normal: May suggest autonomic imbalance or training stress\n"
        content += "• Very low values: Consider further evaluation\n\n"
        
        content += "INTERPRETATION GUIDELINES:\n"
        content += "-" * 28 + "\n"
        content += "• Individual trends over time are more important than single values\n"
        content += "• Values can vary with age, fitness, health status, and stress\n"
        content += "• Morning resting measurements are most reliable\n"
        content += "• Multiple measurements provide better assessment\n"
        content += "• Context matters: sleep, stress, illness, training affect HRV\n\n"
        
        content += "RECOMMENDATIONS BY RANGE:\n"
        content += "-" * 26 + "\n"
        content += "• Normal values: Maintain current health practices\n"
        content += "• High values: Continue current training and lifestyle\n"
        content += "• Low values: Consider lifestyle modifications, stress reduction\n"
        content += "• Very low values: Consult healthcare provider if persistent\n\n"
        
        return content
    
    def _generate_citations_content(self) -> str:
        """Generate detailed scientific citations."""
        content = "SCIENTIFIC CITATIONS AND REFERENCES\n"
        content += "=" * 38 + "\n\n"
        content += "All reference ranges are based on peer-reviewed scientific literature.\n"
        content += "The following studies provide the evidence base for the normal values:\n\n"
        
        # Collect unique citations
        citations = set()
        all_metrics = hrv_reference_ranges.get_all_metrics()
        
        for metric_key in all_metrics:
            citation_info = hrv_reference_ranges.get_citation_info(metric_key)
            if citation_info:
                citations.add((citation_info['citation'], citation_info['doi_pmid']))
        
        # Display citations
        for i, (citation, doi_pmid) in enumerate(sorted(citations), 1):
            content += f"{i}. {citation}\n"
            content += f"   {doi_pmid}\n\n"
        
        content += "ADDITIONAL NOTES:\n"
        content += "-" * 18 + "\n"
        content += "• Reference ranges are specifically for healthy females aged 24-45 years\n"
        content += "• Values may vary with different populations, age groups, and health status\n"
        content += "• 5-minute recordings are standard for clinical HRV assessment\n"
        content += "• 24-hour recordings provide additional long-term variability information\n"
        content += "• Individual variation is normal; trends over time are most important\n\n"
        
        content += "MEASUREMENT CONDITIONS:\n"
        content += "-" * 24 + "\n"
        content += "• Supine resting position\n"
        content += "• Controlled breathing (spontaneous or paced)\n"
        content += "• Quiet environment\n"
        content += "• ECG or high-quality pulse recording\n"
        content += "• Artifact-free data segments\n"
        content += "• Standardized time of day (morning preferred)\n\n"
        
        return content
    
    def _handle_link_click(self, event):
        """Handle clicks on links in citation text."""
        # Simplified link handling - in a full implementation,
        # this would parse DOI/PMID links and open them
        try:
            text_widget = event.widget
            index = text_widget.index(tk.CURRENT)
            line_text = text_widget.get(f"{index.split('.')[0]}.0", f"{index.split('.')[0]}.end")
            
            # Look for DOI or PMID
            if "DOI:" in line_text or "PMID:" in line_text:
                # Extract and open link (simplified)
                if "10.1371/journal.pone.0118308" in line_text:
                    webbrowser.open("https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118308")
                elif "10.3389/fpubh.2017.00258" in line_text:
                    webbrowser.open("https://www.frontiersin.org/articles/10.3389/fpubh.2017.00258/full")
                
        except Exception as e:
            logger.warning(f"Error handling link click: {e}")
    
    def export_citations(self):
        """Export citations to text file."""
        try:
            from tkinter import filedialog
            
            filename = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
                title="Save HRV Citations"
            )
            
            if filename:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(self._generate_full_export_content())
                
                tk.messagebox.showinfo(
                    "Export Complete",
                    f"HRV reference citations exported to:\n{filename}"
                )
                
        except Exception as e:
            logger.error(f"Error exporting citations: {e}")
            tk.messagebox.showerror(
                "Export Error",
                f"Failed to export citations:\n{e}"
            )
    
    def _generate_full_export_content(self) -> str:
        """Generate complete export content."""
        content = "HRV REFERENCE RANGES - COMPLETE SCIENTIFIC DOCUMENTATION\n"
        content += "=" * 65 + "\n\n"
        content += f"Generated: {tk.datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        content += "Source: Enhanced HRV Analysis System\n"
        content += "Population: Healthy females, 24-45 years\n\n"
        
        content += self._generate_time_domain_content() + "\n"
        content += self._generate_frequency_domain_content() + "\n"
        content += self._generate_nonlinear_domain_content() + "\n"
        content += self._generate_clinical_thresholds_content() + "\n"
        content += self._generate_citations_content() + "\n"
        
        return content
    
    def close_window(self):
        """Close the citation display window."""
        if self.window:
            self.window.destroy()
    
    def show(self):
        """Show the window."""
        if self.window:
            self.window.deiconify()
            self.window.lift()
            self.window.focus_force()


def show_hrv_citations(parent=None):
    """
    Convenience function to show HRV citations window.
    
    Args:
        parent: Parent window (optional)
    """
    try:
        window = HRVCitationDisplayWindow(parent)
        window.show()
        logger.info("HRV citations window displayed")
    except Exception as e:
        logger.error(f"Error displaying HRV citations: {e}")
        if parent:
            tk.messagebox.showerror(
                "Citation Display Error",
                f"Failed to display HRV citations:\n{e}"
            )


if __name__ == "__main__":
    # Test the citation display
    root = tk.Tk()
    root.withdraw()
    show_hrv_citations()
    root.mainloop() 