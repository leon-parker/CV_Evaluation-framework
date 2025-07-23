"""
Simple GUI for Autism-Aware Cartoon Pipeline
Basic interface for easy storyboard generation
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
from PIL import Image, ImageTk
import os

class AutismPipelineGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Autism-Friendly Storyboard Generator")
        self.root.geometry("800x600")
        
        # Pipeline placeholder (will be initialized with actual pipeline)
        self.pipeline = None
        self.current_image = None
        
        self.setup_ui()
        
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="Autism-Friendly Storyboard Generator", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=10)
        
        # Model selection
        ttk.Label(main_frame, text="Model Path:").grid(row=1, column=0, sticky=tk.W)
        self.model_path_var = tk.StringVar(value="models/realcartoonxl_v7.safetensors")
        model_entry = ttk.Entry(main_frame, textvariable=self.model_path_var, width=50)
        model_entry.grid(row=1, column=1, padx=5)
        ttk.Button(main_frame, text="Browse", command=self.browse_model).grid(row=1, column=2)
        
        # Initialize button
        self.init_button = ttk.Button(main_frame, text="Initialize Pipeline", 
                                     command=self.initialize_pipeline)
        self.init_button.grid(row=2, column=0, columnspan=3, pady=10)
        
        # Prompt entry
        ttk.Label(main_frame, text="Prompt:").grid(row=3, column=0, sticky=tk.W)
        self.prompt_text = tk.Text(main_frame, height=3, width=60)
        self.prompt_text.grid(row=3, column=1, columnspan=2, pady=5)
        self.prompt_text.insert('1.0', "young boy reading a book, simple bedroom, soft colors, cartoon style")
        
        # Generation options
        options_frame = ttk.LabelFrame(main_frame, text="Options", padding="5")
        options_frame.grid(row=4, column=0, columnspan=3, pady=10, sticky=(tk.W, tk.E))
        
        self.enable_autism_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Enable Autism Scoring", 
                       variable=self.enable_autism_var).grid(row=0, column=0, sticky=tk.W)
        
        self.use_ip_adapter_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Use IP-Adapter", 
                       variable=self.use_ip_adapter_var).grid(row=0, column=1, sticky=tk.W)
        
        ttk.Label(options_frame, text="Images to generate:").grid(row=1, column=0, sticky=tk.W)
        self.num_images_var = tk.IntVar(value=3)
        ttk.Spinbox(options_frame, from_=1, to=5, textvariable=self.num_images_var, 
                   width=10).grid(row=1, column=1, sticky=tk.W)
        
        # Generate button
        self.generate_button = ttk.Button(main_frame, text="Generate Image", 
                                         command=self.generate_image, state='disabled')
        self.generate_button.grid(row=5, column=0, columnspan=3, pady=10)
        
        # Image display
        self.image_label = ttk.Label(main_frame, text="Generated image will appear here")
        self.image_label.grid(row=6, column=0, columnspan=3, pady=10)
        
        # Results
        results_frame = ttk.LabelFrame(main_frame, text="Results", padding="5")
        results_frame.grid(row=7, column=0, columnspan=3, pady=10, sticky=(tk.W, tk.E))
        
        self.results_text = tk.Text(results_frame, height=8, width=70)
        self.results_text.grid(row=0, column=0)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=8, column=0, columnspan=3, sticky=(tk.W, tk.E))
    
    def browse_model(self):
        filename = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("SafeTensors", "*.safetensors"), ("All Files", "*.*")]
        )
        if filename:
            self.model_path_var.set(filename)
    
    def initialize_pipeline(self):
        self.status_var.set("Initializing pipeline...")
        self.init_button.config(state='disabled')
        
        def init_thread():
            try:
                # Import and initialize the pipeline
                from autism_integrated_pipeline import AutismIntegratedCartoonPipeline
                
                model_path = self.model_path_var.get()
                self.pipeline = AutismIntegratedCartoonPipeline(
                    model_path=model_path,
                    enable_autism_scoring=True
                )
                
                if self.pipeline.available:
                    self.root.after(0, self.pipeline_initialized)
                else:
                    self.root.after(0, lambda: self.show_error("Pipeline initialization failed"))
                    
            except Exception as e:
                self.root.after(0, lambda: self.show_error(f"Error: {str(e)}"))
        
        threading.Thread(target=init_thread).start()
    
    def pipeline_initialized(self):
        self.status_var.set("Pipeline initialized successfully!")
        self.generate_button.config(state='normal')
        self.init_button.config(text="✓ Pipeline Ready", state='disabled')
    
    def generate_image(self):
        if not self.pipeline:
            return
        
        prompt = self.prompt_text.get('1.0', 'end-1c')
        if not prompt.strip():
            messagebox.showwarning("Warning", "Please enter a prompt")
            return
        
        self.status_var.set("Generating image...")
        self.generate_button.config(state='disabled')
        self.results_text.delete('1.0', tk.END)
        
        def generate_thread():
            try:
                result = self.pipeline.generate_single_image(
                    prompt=prompt,
                    num_images=self.num_images_var.get(),
                    use_ip_adapter=self.use_ip_adapter_var.get()
                )
                
                if result:
                    self.root.after(0, lambda: self.display_result(result))
                else:
                    self.root.after(0, lambda: self.show_error("Generation failed"))
                    
            except Exception as e:
                self.root.after(0, lambda: self.show_error(f"Generation error: {str(e)}"))
        
        threading.Thread(target=generate_thread).start()
    
    def display_result(self, result):
        # Display image
        image = result["image"]
        # Resize for display
        display_size = (512, 512)
        image_display = image.copy()
        image_display.thumbnail(display_size, Image.Resampling.LANCZOS)
        
        photo = ImageTk.PhotoImage(image_display)
        self.image_label.config(image=photo, text="")
        self.image_label.image = photo  # Keep reference
        
        self.current_image = image
        
        # Display results
        results_text = f"Generation Complete!\n"
        results_text += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        results_text += f"Quality Score: {result['score']:.3f}\n"
        
        if "autism_score" in result:
            results_text += f"Autism Score: {result['autism_score']:.3f}\n"
            results_text += f"Autism Grade: {result['autism_grade']}\n"
            
            # Add recommendations if available
            if "autism_analysis" in result:
                analysis = result["autism_analysis"]
                results_text += f"\nKey Metrics:\n"
                results_text += f"• Person Count: {analysis['person_count']['count']}\n"
                results_text += f"• Background Simplicity: {analysis['background_simplicity']['score']:.2f}\n"
                results_text += f"• Color Appropriateness: {analysis['color_appropriateness']['score']:.2f}\n"
                
                if analysis['recommendations']:
                    results_text += f"\nRecommendations:\n"
                    for rec in analysis['recommendations'][:3]:
                        results_text += f"• {rec}\n"
        
        self.results_text.insert('1.0', results_text)
        
        self.status_var.set("Generation complete! Right-click image to save.")
        self.generate_button.config(state='normal')
        
        # Add save functionality
        self.image_label.bind("<Button-3>", self.save_image)
    
    def save_image(self, event):
        if not self.current_image:
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All Files", "*.*")]
        )
        if filename:
            self.current_image.save(filename)
            self.status_var.set(f"Image saved to {filename}")
    
    def show_error(self, message):
        messagebox.showerror("Error", message)
        self.status_var.set("Error occurred")
        self.generate_button.config(state='normal')
        self.init_button.config(state='normal')


def main():
    root = tk.Tk()
    app = AutismPipelineGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()