"""
HRV Metrics Scientific Explanations Module

This module provides comprehensive scientific explanations for Heart Rate Variability
(HRV) metrics based on peer-reviewed research and clinical literature.

The explanations are designed for both clinical and research applications,
providing physiological basis, clinical significance, and interpretation guidelines
for each HRV parameter.

Author: AI Assistant
Date: 2025-01-14
Integration: Enhanced HRV Analysis System
"""

import tkinter as tk
from tkinter import ttk, scrolledtext
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class HRVMetricsExplanationWindow:
    """Window for displaying comprehensive HRV metrics explanations."""
    
    def __init__(self, parent, metrics_data: Optional[Dict[str, Any]] = None):
        """Initialize the HRV metrics explanation window."""
        self.parent = parent
        self.metrics_data = metrics_data or {}
        
        # Create the explanation window
        self.window = tk.Toplevel(parent)
        self.window.title("HRV Metrics - Scientific Explanations")
        self.window.geometry("1000x700")
        
        # Configure window icon
        try:
            self.window.iconbitmap(str(parent.winfo_toplevel().iconbitmap()))
        except:
            pass
        
        self._setup_ui()
        
        # Center the window
        self.window.transient(parent)
        self.window.grab_set()
        self.window.update_idletasks()
        
        # Center on parent
        x = parent.winfo_rootx() + (parent.winfo_width() // 2) - (1000 // 2)
        y = parent.winfo_rooty() + (parent.winfo_height() // 2) - (700 // 2)
        self.window.geometry(f"1000x700+{x}+{y}")
        
    def _setup_ui(self):
        """Setup the user interface."""
        # Main frame
        main_frame = ttk.Frame(self.window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title_label = ttk.Label(
            main_frame,
            text="Heart Rate Variability (HRV) Metrics - Scientific Explanations",
            font=('Arial', 16, 'bold')
        )
        title_label.pack(pady=(0, 10))
        
        # Subtitle
        subtitle_label = ttk.Label(
            main_frame,
            text="Based on peer-reviewed research and clinical literature",
            font=('Arial', 10, 'italic')
        )
        subtitle_label.pack(pady=(0, 20))
        
        # Create notebook for different categories
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs
        self._create_overview_tab()
        self._create_time_domain_tab()
        self._create_frequency_domain_tab()
        self._create_nonlinear_tab()
        self._create_clinical_significance_tab()
        self._create_interpretation_tab()
        
        # Close button
        close_frame = ttk.Frame(main_frame)
        close_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(
            close_frame,
            text="Close",
            command=self.window.destroy
        ).pack(side=tk.RIGHT)
        
    def _create_overview_tab(self):
        """Create the overview tab."""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Overview")
        
        text_widget = scrolledtext.ScrolledText(frame, wrap=tk.WORD, font=('Arial', 11))
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        overview_text = """HEART RATE VARIABILITY (HRV) - SCIENTIFIC OVERVIEW

Heart Rate Variability represents the variation in time intervals between consecutive heartbeats, reflecting the dynamic interplay between the sympathetic and parasympathetic branches of the autonomic nervous system.

PHYSIOLOGICAL BASIS:
• HRV emerges from complex interactions between multiple physiological control systems
• Reflects the heart's ability to respond to internal and external stimuli
• Provides non-invasive assessment of autonomic nervous system function
• Indicates cardiovascular health, stress resilience, and adaptability

CLINICAL SIGNIFICANCE:
• Higher HRV generally indicates better cardiovascular health and autonomic function
• Reduced HRV is associated with increased mortality risk, cardiovascular disease, and various pathological conditions
• Age-adjusted HRV values are important for proper interpretation
• Individual baselines and temporal changes are more informative than single measurements

MEASUREMENT DOMAINS:
1. Time Domain: Statistical measures of RR intervals over time
2. Frequency Domain: Power spectral analysis of heart rate oscillations
3. Nonlinear Methods: Complexity and fractal analysis of heart rhythms

AUTONOMIC NERVOUS SYSTEM INFLUENCE:
• Parasympathetic (Vagal) Activity: Promotes heart rate variability, especially at higher frequencies
• Sympathetic Activity: Influences longer-term variability and overall autonomic balance
• Complex Interactions: Both systems work together in healthy individuals

CLINICAL APPLICATIONS:
• Cardiovascular risk assessment
• Autonomic neuropathy detection (especially in diabetes)
• Stress and fatigue monitoring
• Training load optimization in athletes
• Mental health assessment and intervention monitoring

IMPORTANT CONSIDERATIONS:
• Recording conditions significantly affect measurements
• Breathing patterns influence HRV metrics
• Medication effects should be considered
• Population-specific reference values are essential"""
        
        text_widget.insert(tk.END, overview_text)
        text_widget.config(state=tk.DISABLED)
        
    def _create_time_domain_tab(self):
        """Create the time domain metrics tab."""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Time Domain")
        
        text_widget = scrolledtext.ScrolledText(frame, wrap=tk.WORD, font=('Arial', 11))
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        time_domain_text = """TIME DOMAIN HRV METRICS - SCIENTIFIC EXPLANATIONS

Time domain measures quantify the amount of variance in RR intervals using statistical methods. These are the simplest and most widely used HRV metrics.

═══════════════════════════════════════════════════════════════════════════════

SDNN (Standard Deviation of NN Intervals)
• DEFINITION: Standard deviation of all normal RR intervals over the recording period
• UNITS: Milliseconds (ms)
• PHYSIOLOGICAL BASIS: Reflects overall heart rate variability from all sources
• CLINICAL SIGNIFICANCE:
  - Global measure of autonomic function
  - Incorporates both parasympathetic and sympathetic influences
  - Stronger predictor of mortality than other time domain measures
  - Values <50ms indicate severely compromised autonomic function
• INTERPRETATION:
  - Higher values = better overall autonomic function
  - Age-related decline is normal (approximately 1ms per year after age 40)
  - Exercise training can improve SDNN values
• NORMAL VALUES (24-hour recordings):
  - Healthy adults: 120-200ms
  - Elderly (>65 years): 80-150ms
  - Athletes: Often >150ms

═══════════════════════════════════════════════════════════════════════════════

RMSSD (Root Mean Square of Successive Differences)
• DEFINITION: Square root of the mean squared differences between successive RR intervals
• UNITS: Milliseconds (ms)
• PHYSIOLOGICAL BASIS: Primarily reflects parasympathetic (vagal) activity
• CLINICAL SIGNIFICANCE:
  - Best time domain measure of vagal tone
  - Correlates strongly with respiratory sinus arrhythmia
  - Less affected by breathing pattern than frequency domain measures
  - Excellent marker of recovery and stress resilience
• INTERPRETATION:
  - Higher values = stronger parasympathetic activity
  - Responds quickly to stress and recovery
  - Useful for day-to-day monitoring in athletes
• NORMAL VALUES (5-minute recordings):
  - Healthy young adults: 30-60ms
  - Healthy middle-aged: 20-40ms
  - Athletes: Often >40ms

═══════════════════════════════════════════════════════════════════════════════

pNN50 (Percentage of NN50)
• DEFINITION: Percentage of successive RR intervals that differ by more than 50ms
• UNITS: Percentage (%)
• PHYSIOLOGICAL BASIS: Reflects parasympathetic influence on heart rate
• CLINICAL SIGNIFICANCE:
  - Correlates with RMSSD but less sensitive at low HRV levels
  - Useful for detecting autonomic neuropathy
  - Age-dependent with significant decline after age 50
• INTERPRETATION:
  - Higher values = better parasympathetic function
  - Values <3% may indicate autonomic dysfunction
• NORMAL VALUES (24-hour recordings):
  - Young healthy adults: 10-25%
  - Middle-aged adults: 5-15%
  - Elderly: 2-10%

═══════════════════════════════════════════════════════════════════════════════

TRIANGULAR INDEX (TINN)
• DEFINITION: Baseline width of the RR interval histogram
• UNITS: Milliseconds (ms)
• PHYSIOLOGICAL BASIS: Geometric measure of overall variability
• CLINICAL SIGNIFICANCE:
  - Less sensitive to artifacts than statistical measures
  - Good correlation with SDNN
  - Requires longer recordings (>20 minutes)
• INTERPRETATION:
  - Higher values = better overall HRV
• NORMAL VALUES (24-hour recordings):
  - Healthy adults: 20-50ms
  - Values <15ms indicate poor autonomic function

═══════════════════════════════════════════════════════════════════════════════

CLINICAL APPLICATIONS OF TIME DOMAIN MEASURES:

1. CARDIOVASCULAR RISK ASSESSMENT:
   - SDNN <70ms (24-hour) indicates high risk
   - RMSSD <15ms (short-term) suggests autonomic dysfunction

2. AUTONOMIC NEUROPATHY SCREENING:
   - Particularly useful in diabetes mellitus
   - pNN50 <0.5% highly suggestive of neuropathy

3. TRAINING MONITORING:
   - Daily RMSSD tracking for overtraining prevention
   - Recovery assessment in athletes

4. STRESS ASSESSMENT:
   - Acute stress reduces RMSSD significantly
   - Chronic stress leads to overall SDNN reduction

LIMITATIONS:
• Cannot distinguish between sympathetic and parasympathetic contributions
• Affected by recording length and conditions
• Require normal sinus rhythm for accurate measurement"""
        
        text_widget.insert(tk.END, time_domain_text)
        text_widget.config(state=tk.DISABLED)
        
    def _create_frequency_domain_tab(self):
        """Create the frequency domain metrics tab."""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Frequency Domain")
        
        text_widget = scrolledtext.ScrolledText(frame, wrap=tk.WORD, font=('Arial', 11))
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        frequency_domain_text = """FREQUENCY DOMAIN HRV METRICS - SCIENTIFIC EXPLANATIONS

Frequency domain analysis decomposes HRV into component oscillations at different frequencies, providing insights into specific physiological mechanisms underlying heart rate variability.

═══════════════════════════════════════════════════════════════════════════════

HIGH FREQUENCY (HF) POWER
• DEFINITION: Power in the 0.15-0.40 Hz frequency range
• UNITS: ms² (absolute) or normalized units (nu)
• PHYSIOLOGICAL BASIS:
  - Primarily reflects parasympathetic (vagal) activity
  - Corresponds to respiratory sinus arrhythmia (RSA)
  - Modulated by breathing frequency and depth
• CLINICAL SIGNIFICANCE:
  - Direct marker of vagal tone
  - Reduced in stress, anxiety, and cardiovascular disease
  - Increases with relaxation and deep breathing
  - Strong predictor of cardiovascular outcomes
• INTERPRETATION:
  - Higher values = stronger parasympathetic activity
  - Typically peaks during sleep and rest
  - Decreases with age and pathological conditions
• NORMAL VALUES (5-minute recordings):
  - Young adults: 500-2000 ms²
  - Values <100 ms² indicate reduced vagal tone

═══════════════════════════════════════════════════════════════════════════════

LOW FREQUENCY (LF) POWER
• DEFINITION: Power in the 0.04-0.15 Hz frequency range
• UNITS: ms² (absolute) or normalized units (nu)
• PHYSIOLOGICAL BASIS:
  - Reflects both sympathetic and parasympathetic influences
  - Associated with baroreflex activity
  - Includes Mayer wave oscillations (~0.1 Hz)
• CLINICAL SIGNIFICANCE:
  - CONTROVERSIAL: Not a pure marker of sympathetic activity
  - Primarily reflects baroreflex function during rest
  - Reduced in heart failure and autonomic dysfunction
• INTERPRETATION:
  - Higher values may indicate active autonomic regulation
  - Context-dependent interpretation required
  - Should not be used alone as sympathetic marker
• NORMAL VALUES (5-minute recordings):
  - Young adults: 300-1500 ms²

═══════════════════════════════════════════════════════════════════════════════

VERY LOW FREQUENCY (VLF) POWER
• DEFINITION: Power in the 0.003-0.04 Hz frequency range
• UNITS: ms² (absolute)
• PHYSIOLOGICAL BASIS:
  - Generated by intrinsic cardiac nervous system
  - Reflects long-term regulatory mechanisms
  - Associated with thermoregulation, hormonal factors
• CLINICAL SIGNIFICANCE:
  - Strongest predictor of mortality among frequency measures
  - Reduced in heart failure, diabetes, and aging
  - Associated with inflammation markers
• INTERPRETATION:
  - Higher values = better long-term regulatory capacity
  - Most predictive of cardiovascular outcomes
• NORMAL VALUES (24-hour recordings):
  - Healthy adults: 500-3000 ms²

═══════════════════════════════════════════════════════════════════════════════

ULTRA LOW FREQUENCY (ULF) POWER
• DEFINITION: Power in the <0.003 Hz frequency range
• UNITS: ms² (absolute)
• PHYSIOLOGICAL BASIS:
  - Reflects circadian rhythms
  - Associated with core body temperature regulation
  - Influenced by sleep-wake cycles
• CLINICAL SIGNIFICANCE:
  - Requires 24-hour recordings
  - Disrupted in shift work and circadian disorders
• NORMAL VALUES (24-hour recordings):
  - Healthy adults: 1000-5000 ms²

═══════════════════════════════════════════════════════════════════════════════

LF/HF RATIO
• DEFINITION: Ratio of low frequency to high frequency power
• UNITS: Dimensionless ratio
• PHYSIOLOGICAL BASIS:
  - CONTROVERSIAL: Originally proposed as sympatho-vagal balance
  - Actually reflects relative distribution of autonomic influences
• CLINICAL SIGNIFICANCE:
  - CAUTION: Should not be interpreted as sympathetic/parasympathetic ratio
  - Influenced by multiple factors including breathing
  - May indicate autonomic balance shifts
• INTERPRETATION:
  - Higher values may suggest relative sympathetic predominance
  - Lower values may indicate relative parasympathetic predominance
  - Context and individual baseline important
• NORMAL VALUES:
  - Resting adults: 0.5-2.0
  - Stress increases ratio
  - Exercise training may reduce ratio

═══════════════════════════════════════════════════════════════════════════════

TOTAL POWER
• DEFINITION: Sum of all frequency components
• UNITS: ms² (absolute)
• PHYSIOLOGICAL BASIS: Overall variability across all frequencies
• CLINICAL SIGNIFICANCE:
  - Global measure of autonomic function
  - Strongly correlates with time domain measures
• INTERPRETATION: Higher values = better overall HRV

═══════════════════════════════════════════════════════════════════════════════

CLINICAL APPLICATIONS:

1. AUTONOMIC ASSESSMENT:
   - HF power for parasympathetic function
   - VLF power for overall regulatory capacity
   - Total power for global autonomic health

2. CARDIOVASCULAR RISK:
   - VLF power most predictive of mortality
   - Reduced HF power indicates increased risk

3. STRESS MONITORING:
   - HF power decreases with acute stress
   - LF/HF ratio changes not reliable for chronic stress

4. RESPIRATORY INFLUENCES:
   - Slow breathing (<8 breaths/min) shifts power to LF band
   - Paced breathing protocols standardize measurements

METHODOLOGICAL CONSIDERATIONS:
• Recording length affects frequency resolution
• Breathing pattern significantly influences results
• Stationarity of signal important for accurate analysis
• Artifact removal critical for valid results"""
        
        text_widget.insert(tk.END, frequency_domain_text)
        text_widget.config(state=tk.DISABLED)
        
    def _create_nonlinear_tab(self):
        """Create the nonlinear metrics tab."""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Nonlinear")
        
        text_widget = scrolledtext.ScrolledText(frame, wrap=tk.WORD, font=('Arial', 11))
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        nonlinear_text = """NONLINEAR HRV METRICS - SCIENTIFIC EXPLANATIONS

Nonlinear analysis methods capture the complex, non-random patterns in heart rate dynamics that may not be detected by traditional time and frequency domain measures.

═══════════════════════════════════════════════════════════════════════════════

POINCARÉ PLOT ANALYSIS

SD1 (Short-term Variability)
• DEFINITION: Standard deviation perpendicular to the line of identity in Poincaré plot
• UNITS: Milliseconds (ms)
• PHYSIOLOGICAL BASIS:
  - Reflects beat-to-beat variability
  - Primarily parasympathetic influence
  - Corresponds to RMSSD (SD1 = RMSSD/√2)
• CLINICAL SIGNIFICANCE:
  - Marker of short-term heart rate regulation
  - Sensitive to autonomic dysfunction
  - Reduced in stress and pathological conditions
• INTERPRETATION:
  - Higher values = better short-term variability
  - Responds rapidly to autonomic changes
• NORMAL VALUES:
  - Healthy young adults: 20-50ms
  - Decreases with age and cardiovascular disease

SD2 (Long-term Variability)
• DEFINITION: Standard deviation along the line of identity in Poincaré plot
• UNITS: Milliseconds (ms)
• PHYSIOLOGICAL BASIS:
  - Reflects long-term heart rate variability
  - Influenced by both sympathetic and parasympathetic systems
  - Related to SDNN (SD2² = 2×SDNN² - SD1²)
• CLINICAL SIGNIFICANCE:
  - Marker of overall autonomic function
  - Predictor of cardiovascular outcomes
• INTERPRETATION:
  - Higher values = better long-term regulation
• NORMAL VALUES:
  - Healthy adults: 50-200ms

SD1/SD2 Ratio
• DEFINITION: Ratio of short-term to long-term variability
• PHYSIOLOGICAL BASIS:
  - Indicates relative contribution of different regulatory mechanisms
  - Values approach 1 suggest random variability
  - Lower values indicate more complex regulation
• CLINICAL SIGNIFICANCE:
  - May distinguish healthy from pathological patterns
  - Altered in various cardiac conditions
• NORMAL VALUES:
  - Healthy adults: 0.2-0.5

═══════════════════════════════════════════════════════════════════════════════

DETRENDED FLUCTUATION ANALYSIS (DFA)

α1 (Short-term Scaling Exponent)
• DEFINITION: Scaling exponent for short-term correlations (4-11 beats)
• PHYSIOLOGICAL BASIS:
  - Reflects short-term fractal properties
  - Related to parasympathetic activity
• CLINICAL SIGNIFICANCE:
  - Altered in cardiovascular disease
  - Predictor of mortality risk
• INTERPRETATION:
  - Values around 1.0-1.5 indicate healthy complexity
  - Values <0.5 or >1.5 may indicate pathology
• NORMAL VALUES:
  - Healthy adults: 0.9-1.3

α2 (Long-term Scaling Exponent)
• DEFINITION: Scaling exponent for long-term correlations (>11 beats)
• PHYSIOLOGICAL BASIS:
  - Reflects long-term fractal organization
  - Associated with overall cardiac regulation
• CLINICAL SIGNIFICANCE:
  - Strong predictor of cardiovascular events
  - Altered before clinical symptoms appear
• NORMAL VALUES:
  - Healthy adults: 0.8-1.2

═══════════════════════════════════════════════════════════════════════════════

ENTROPY MEASURES

Sample Entropy (SampEn)
• DEFINITION: Measure of signal regularity and complexity
• PHYSIOLOGICAL BASIS:
  - Higher values indicate greater complexity
  - Reflects non-linear dynamics of heart rate control
• CLINICAL SIGNIFICANCE:
  - Reduced in pathological conditions
  - Independent predictor of mortality
• INTERPRETATION:
  - Higher values = healthier, more complex regulation
  - Lower values suggest rigid, predictable patterns
• NORMAL VALUES:
  - Healthy adults: 1.5-2.5

Approximate Entropy (ApEn)
• DEFINITION: Measure of time series regularity
• PHYSIOLOGICAL BASIS:
  - Quantifies randomness in heart rate patterns
  - Related to system complexity
• CLINICAL SIGNIFICANCE:
  - Decreased in heart failure and aging
  - Useful for risk stratification

═══════════════════════════════════════════════════════════════════════════════

RECURRENCE QUANTIFICATION ANALYSIS (RQA)

Determinism (DET)
• DEFINITION: Percentage of recurrent points forming diagonal lines
• PHYSIOLOGICAL BASIS:
  - Reflects deterministic behavior in heart rate dynamics
  - Higher values indicate more predictable patterns
• CLINICAL SIGNIFICANCE:
  - Altered in various pathological conditions
  - May detect subtle changes in cardiac regulation

Entropy (ENTR)
• DEFINITION: Shannon entropy of diagonal line lengths
• PHYSIOLOGICAL BASIS:
  - Reflects complexity of deterministic structures
• CLINICAL SIGNIFICANCE:
  - Provides information about system complexity

═══════════════════════════════════════════════════════════════════════════════

CLINICAL APPLICATIONS OF NONLINEAR MEASURES:

1. CARDIOVASCULAR RISK ASSESSMENT:
   - DFA α1 <0.75 indicates high mortality risk
   - Sample Entropy <1.0 suggests increased risk

2. AUTONOMIC DYSFUNCTION DETECTION:
   - SD1 highly sensitive to parasympathetic changes
   - Entropy measures detect subtle regulatory changes

3. HEART FAILURE MONITORING:
   - Multiple nonlinear measures altered in heart failure
   - May provide earlier detection than linear measures

4. AGING STUDIES:
   - Complexity measures decline with healthy aging
   - May distinguish normal aging from pathological changes

5. PHARMACOLOGICAL EFFECTS:
   - Nonlinear measures sensitive to medication effects
   - Useful for monitoring therapeutic interventions

═══════════════════════════════════════════════════════════════════════════════

ADVANTAGES OF NONLINEAR ANALYSIS:
• Captures information not available in linear measures
• May be more sensitive to subtle physiological changes
• Provides insights into the complexity of cardiac regulation
• Less affected by non-stationarity in some cases

LIMITATIONS:
• Requires longer recordings for reliable estimates
• Computational complexity and interpretation challenges
• Normal values less well established
• May be sensitive to artifacts and noise"""
        
        text_widget.insert(tk.END, nonlinear_text)
        text_widget.config(state=tk.DISABLED)
        
    def _create_clinical_significance_tab(self):
        """Create the clinical significance tab."""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Clinical Significance")
        
        text_widget = scrolledtext.ScrolledText(frame, wrap=tk.WORD, font=('Arial', 11))
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        clinical_text = """CLINICAL SIGNIFICANCE OF HRV METRICS

Heart Rate Variability has emerged as a powerful biomarker for autonomic function, cardiovascular health, and overall physiological regulation. This section outlines the clinical significance and applications of HRV measurements.

═══════════════════════════════════════════════════════════════════════════════

CARDIOVASCULAR DISEASE AND MORTALITY PREDICTION

Post-Myocardial Infarction:
• SDNN <70ms (24-hour) indicates 3-5 fold increased mortality risk
• RMSSD <15ms predicts sudden cardiac death
• VLF power most predictive of long-term outcomes
• Combined HRV measures improve risk stratification beyond traditional factors

Heart Failure:
• Progressive HRV reduction correlates with disease severity
• SDNN <100ms indicates poor prognosis
• Nonlinear measures may provide earlier detection
• HRV-guided therapy shows promise in management

Coronary Artery Disease:
• Reduced HRV precedes clinical manifestations
• RMSSD <20ms suggests significant autonomic dysfunction
• Exercise-induced HRV changes predict cardiac events
• HRV recovery after exercise indicates prognosis

═══════════════════════════════════════════════════════════════════════════════

AUTONOMIC NEUROPATHY ASSESSMENT

Diabetic Autonomic Neuropathy:
• HRV reduction may precede clinical symptoms by years
• pNN50 <0.5% highly suggestive of neuropathy
• RMSSD <15ms indicates established dysfunction
• Progression monitoring through serial measurements

Age-Related Autonomic Decline:
• SDNN decreases approximately 1ms per year after age 40
• HF power shows steeper decline than time domain measures
• Complexity measures (entropy, DFA) decline with aging
• Important to use age-adjusted reference values

Other Neuropathies:
• Chronic kidney disease: Progressive HRV reduction
• Multiple sclerosis: Altered autonomic regulation patterns
• Parkinson's disease: Reduced complexity measures

═══════════════════════════════════════════════════════════════════════════════

MENTAL HEALTH AND STRESS ASSESSMENT

Depression and Anxiety:
• Reduced RMSSD and HF power in major depression
• Anxiety disorders show altered LF/HF patterns
• HRV may predict treatment response
• Biofeedback using HRV shows therapeutic benefits

Post-Traumatic Stress Disorder (PTSD):
• Significant reduction in multiple HRV parameters
• Altered stress response patterns
• HRV training shows promise as intervention

Chronic Stress:
• Progressive reduction in overall HRV
• Altered circadian patterns of autonomic function
• Recovery monitoring through HRV normalization

═══════════════════════════════════════════════════════════════════════════════

SLEEP DISORDERS AND CIRCADIAN RHYTHMS

Sleep Apnea:
• Characteristic alterations in nocturnal HRV patterns
• Reduced RR variability during apneic episodes
• Treatment monitoring through HRV improvement

Insomnia:
• Altered autonomic balance during sleep attempts
• Reduced parasympathetic activity at bedtime
• HRV biofeedback may improve sleep quality

Shift Work Disorders:
• Disrupted circadian HRV patterns
• Altered ULF and VLF components
• Risk assessment for cardiovascular complications

═══════════════════════════════════════════════════════════════════════════════

SPORTS MEDICINE AND PERFORMANCE

Overtraining Syndrome:
• Progressive decline in RMSSD over days to weeks
• Altered nocturnal HRV patterns
• Early detection before performance decline

Training Load Monitoring:
• Daily RMSSD tracking for training prescription
• HRV-guided training improves performance outcomes
• Individual baseline establishment crucial

Recovery Assessment:
• HRV normalization indicates physiological recovery
• Faster recovery associated with better fitness
• Useful for return-to-play decisions after injury

═══════════════════════════════════════════════════════════════════════════════

MEDICATION EFFECTS AND MONITORING

Beta-Blockers:
• Increase RMSSD and HF power
• Reduce exercise-induced HRV changes
• Cardioprotective effects partially mediated through HRV

ACE Inhibitors/ARBs:
• Gradual improvement in HRV parameters
• Correlates with cardiovascular protection
• Useful for monitoring treatment effectiveness

Antiarrhythmic Drugs:
• Complex effects on HRV parameters
• May help predict drug efficacy
• Important for safety monitoring

═══════════════════════════════════════════════════════════════════════════════

PEDIATRIC APPLICATIONS

Fetal Distress:
• Reduced fetal HRV precedes other signs of distress
• Important for obstetric monitoring
• Predictive of neonatal outcomes

ADHD and Autism:
• Altered autonomic regulation patterns
• HRV biofeedback shows therapeutic promise
• Useful for treatment monitoring

Congenital Heart Disease:
• HRV patterns relate to functional capacity
• Risk stratification for surgical interventions
• Long-term outcome prediction

═══════════════════════════════════════════════════════════════════════════════

GERIATRIC APPLICATIONS

Frailty Assessment:
• HRV reduction correlates with frailty scores
• Predictive of falls and functional decline
• Useful for comprehensive geriatric assessment

Cognitive Decline:
• Altered HRV patterns in mild cognitive impairment
• May precede clinical dementia diagnosis
• Potential biomarker for neurodegenerative diseases

═══════════════════════════════════════════════════════════════════════════════

CRITICAL CARE APPLICATIONS

Sepsis and Critical Illness:
• Characteristic HRV alterations in sepsis
• May predict organ failure and mortality
• Useful for early intervention triggering

Mechanical Ventilation:
• HRV patterns predict weaning readiness
• Autonomic recovery assessment
• Guide timing of intervention withdrawal

═══════════════════════════════════════════════════════════════════════════════

POPULATION HEALTH AND EPIDEMIOLOGY

Cardiovascular Risk in Populations:
• Population HRV screening for risk stratification
• Identification of high-risk asymptomatic individuals
• Cost-effective screening strategies

Environmental Health:
• HRV effects of air pollution exposure
• Occupational stress assessment
• Community health monitoring

Research Applications:
• Biomarker in clinical trials
• Mechanism studies of interventions
• Longitudinal health studies"""
        
        text_widget.insert(tk.END, clinical_text)
        text_widget.config(state=tk.DISABLED)
        
    def _create_interpretation_tab(self):
        """Create the interpretation guidelines tab."""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Interpretation Guidelines")
        
        text_widget = scrolledtext.ScrolledText(frame, wrap=tk.WORD, font=('Arial', 11))
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        interpretation_text = """HRV INTERPRETATION GUIDELINES

Proper interpretation of HRV measurements requires understanding of methodology, normal values, and clinical context. This section provides practical guidelines for HRV analysis and interpretation.

═══════════════════════════════════════════════════════════════════════════════

GENERAL INTERPRETATION PRINCIPLES

1. CONTEXT IS CRUCIAL:
• Consider recording conditions (rest, activity, sleep)
• Account for medications and medical conditions
• Use age and gender-appropriate reference values
• Evaluate trends over time rather than single measurements

2. BASELINE ESTABLISHMENT:
• Individual baseline more important than population norms
• Requires multiple measurements under similar conditions
• Consider circadian variations and day-to-day changes
• Establish personal "healthy" ranges

3. MULTIPLE PARAMETER ASSESSMENT:
• No single HRV parameter tells the complete story
• Combine time, frequency, and nonlinear measures when possible
• Look for consistent patterns across parameters
• Consider physiological relationships between measures

═══════════════════════════════════════════════════════════════════════════════

AGE AND GENDER CONSIDERATIONS

Age-Related Changes:
• SDNN decreases approximately 1ms per year after age 40
• HF power shows steeper decline than time domain measures
• Nonlinear measures decline progressively with age
• Use age-adjusted reference values for interpretation

Gender Differences:
• Females typically have higher HF power than males
• Gender differences more pronounced in younger populations
• Hormonal fluctuations affect HRV in females
• Consider menstrual cycle phase in premenopausal women

Pediatric Considerations:
• HRV values generally higher in children and adolescents
• Developmental changes in autonomic maturation
• Different normal ranges for different age groups
• Growth and development affect autonomic function

═══════════════════════════════════════════════════════════════════════════════

RECORDING CONDITIONS AND STANDARDIZATION

Short-term Recordings (5 minutes):
• Controlled conditions essential (supine rest, quiet environment)
• Minimum 5-minute duration for frequency domain analysis
• Breathing pattern significantly affects results
• Multiple recordings improve reliability

Long-term Recordings (24 hours):
• More representative of overall autonomic function
• Include circadian variations and daily activities
• Better for risk stratification and clinical decisions
• Artifact removal more challenging but critical

Positioning Effects:
• Supine: Highest HRV values, maximum parasympathetic activity
• Sitting: Intermediate values, balanced autonomic state
• Standing: Lowest HRV, sympathetic predominance
• Standardize position for serial measurements

═══════════════════════════════════════════════════════════════════════════════

CLINICAL DECISION THRESHOLDS

High-Risk Thresholds (24-hour recordings):
• SDNN <70ms: High mortality risk in post-MI patients
• SDNN <50ms: Very high risk, consider intensive monitoring
• RMSSD <15ms: Significant autonomic dysfunction
• VLF power <9ms²: Increased cardiovascular risk

Autonomic Neuropathy Screening:
• pNN50 <0.5%: Highly suggestive of diabetic neuropathy
• RMSSD <10ms: Established autonomic dysfunction
• Multiple parameters below 5th percentile: Confirmed neuropathy

Athletic Overtraining:
• RMSSD reduction >10ms from baseline: Consider rest day
• RMSSD reduction >20ms from baseline: Likely overreached
• Consecutive days of reduced HRV: Modify training load

═══════════════════════════════════════════════════════════════════════════════

MEDICATION EFFECTS ON HRV

Beta-Blockers:
• Increase RMSSD and HF power (enhanced parasympathetic activity)
• Reduce heart rate and may increase overall HRV
• Effects stabilize after 2-4 weeks of therapy
• Consider when interpreting changes over time

ACE Inhibitors/ARBs:
• Gradual improvement in HRV parameters over months
• Effects may take 3-6 months to fully develop
• Improvement correlates with cardiovascular protection

Antidepressants:
• Tricyclics: Generally reduce HRV
• SSRIs: Variable effects, often minimal impact
• Consider medication effects in interpretation

═══════════════════════════════════════════════════════════════════════════════

COMMON INTERPRETATION PITFALLS

1. OVERINTERPRETATION OF SINGLE MEASUREMENTS:
• HRV has significant day-to-day variability
• Multiple measurements required for reliable assessment
• Trends more important than absolute values

2. IGNORING RECORDING CONDITIONS:
• Breathing pattern profoundly affects HRV
• Physical activity prior to recording influences results
• Time of day affects autonomic balance

3. INAPPROPRIATE REFERENCE VALUES:
• Age and gender matching essential
• Population-specific norms may be required
• Medication effects must be considered

4. LF/HF RATIO MISINTERPRETATION:
• NOT a direct measure of sympatho-vagal balance
• Affected by breathing pattern and multiple factors
• Should not be used alone for clinical decisions

═══════════════════════════════════════════════════════════════════════════════

QUALITY CONTROL AND DATA VALIDATION

Signal Quality Assessment:
• >95% normal sinus beats required for reliable analysis
• Artifact detection and correction essential
• Visual inspection of raw data recommended
• Consider signal quality metrics

Statistical Considerations:
• Normal distribution of HRV parameters often skewed
• Log transformation may be appropriate for some parameters
• Consider confidence intervals and measurement error
• Multiple comparisons adjustments when appropriate

Reproducibility:
• Test-retest reliability varies among parameters
• Time domain measures generally more reproducible
• Frequency domain measures more sensitive to conditions
• Establish measurement protocols for consistency

═══════════════════════════════════════════════════════════════════════════════

CLINICAL REPORTING RECOMMENDATIONS

Essential Information to Include:
1. Recording conditions (duration, position, time of day)
2. Data quality metrics (percentage of normal beats)
3. Age and gender-appropriate reference ranges
4. Multiple HRV parameters when available
5. Clinical context and medication list

Interpretation Statements Should:
• Avoid definitive diagnoses based solely on HRV
• Provide context for findings
• Suggest clinical correlation when appropriate
• Indicate limitations of the measurement
• Recommend follow-up when indicated

Follow-up Recommendations:
• Serial measurements for trend analysis
• Consider 24-hour Holter when short-term results abnormal
• Integrate with other cardiac assessments
• Monitor response to interventions"""
        
        text_widget.insert(tk.END, interpretation_text)
        text_widget.config(state=tk.DISABLED)


def show_hrv_explanations(parent, metrics_data=None):
    """Show the HRV metrics explanation window."""
    try:
        HRVMetricsExplanationWindow(parent, metrics_data)
    except Exception as e:
        logger.error(f"Error showing HRV explanations: {e}")
        tk.messagebox.showerror("Error", f"Could not open HRV explanations: {e}")


# Dictionary for quick metric lookups (can be used programmatically)
HRV_METRICS_QUICK_REFERENCE = {
    'sdnn': {
        'name': 'Standard Deviation of NN Intervals',
        'unit': 'ms',
        'domain': 'Time',
        'physiology': 'Overall HRV from all sources',
        'clinical': 'Global autonomic function marker',
        'normal_range': '120-200ms (24h)',
        'high_risk': '<70ms (24h)'
    },
    'rmssd': {
        'name': 'Root Mean Square of Successive Differences', 
        'unit': 'ms',
        'domain': 'Time',
        'physiology': 'Parasympathetic (vagal) activity',
        'clinical': 'Best time domain marker of vagal tone',
        'normal_range': '30-60ms (5min)',
        'high_risk': '<15ms'
    },
    'pnn50': {
        'name': 'Percentage of NN50',
        'unit': '%',
        'domain': 'Time',
        'physiology': 'Parasympathetic influence',
        'clinical': 'Autonomic neuropathy screening',
        'normal_range': '10-25% (young adults)',
        'high_risk': '<0.5% (neuropathy)'
    },
    'hf_power': {
        'name': 'High Frequency Power',
        'unit': 'ms²',
        'domain': 'Frequency',
        'physiology': 'Parasympathetic activity, respiratory sinus arrhythmia',
        'clinical': 'Direct marker of vagal tone',
        'normal_range': '500-2000ms² (5min)',
        'high_risk': '<100ms²'
    },
    'lf_power': {
        'name': 'Low Frequency Power',
        'unit': 'ms²', 
        'domain': 'Frequency',
        'physiology': 'Mixed autonomic influences, baroreflex activity',
        'clinical': 'NOT pure sympathetic marker',
        'normal_range': '300-1500ms² (5min)',
        'high_risk': 'Context dependent'
    },
    'vlf_power': {
        'name': 'Very Low Frequency Power',
        'unit': 'ms²',
        'domain': 'Frequency', 
        'physiology': 'Intrinsic cardiac regulation, long-term mechanisms',
        'clinical': 'Strongest mortality predictor',
        'normal_range': '500-3000ms² (24h)',
        'high_risk': '<9ms²'
    },
    'lf_hf_ratio': {
        'name': 'LF/HF Ratio',
        'unit': 'ratio',
        'domain': 'Frequency',
        'physiology': 'Relative autonomic influence distribution', 
        'clinical': 'NOT sympatho-vagal balance - use with caution',
        'normal_range': '0.5-2.0 (rest)',
        'high_risk': 'Context dependent'
    },
    'sd1': {
        'name': 'Poincaré SD1',
        'unit': 'ms',
        'domain': 'Nonlinear',
        'physiology': 'Short-term variability, parasympathetic',
        'clinical': 'Beat-to-beat regulation marker',
        'normal_range': '20-50ms',
        'high_risk': '<10ms'
    },
    'sd2': {
        'name': 'Poincaré SD2', 
        'unit': 'ms',
        'domain': 'Nonlinear',
        'physiology': 'Long-term variability, mixed autonomic',
        'clinical': 'Overall autonomic function',
        'normal_range': '50-200ms',
        'high_risk': '<30ms'
    }
} 