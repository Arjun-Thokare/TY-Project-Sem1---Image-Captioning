import os
import cv2
import torch
from PIL import Image, ImageEnhance
from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.textinput import TextInput
from kivy.uix.image import Image as KivyImage
from kivy.uix.spinner import Spinner
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.popup import Popup
from kivy.graphics.texture import Texture
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer

# Lazy load model variables
caption_model = None
feature_extractor = None
tokenizer = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomFileChooser(FileChooserIconView):
    def is_hidden(self, fn):
        """Override to filter out system files."""
        try:
            if super().is_hidden(fn):
                return True
        except Exception:
            return True
        # Exclude specific system files
        return fn.lower() in ('hiberfil.sys', 'pagefile.sys', 'swapfile.sys')

class ImageCaptionApp(App):
    def build(self):
        layout = BoxLayout(orientation='vertical')

        # Input field for image path
        self.image_path_input = TextInput(hint_text='Enter image path', size_hint_y=None, height=50)
        layout.add_widget(self.image_path_input)

        # Button to upload image
        self.upload_button = Button(text="Upload Image", size_hint_y=None, height=50)
        self.upload_button.bind(on_press=self.upload_image)
        layout.add_widget(self.upload_button)

        # Spinner for selecting resolution
        self.resolution_spinner = Spinner(
            text='Select Resolution',
            values=('low', 'medium', 'high'),
            size_hint_y=None,
            height=50
        )
        self.resolution_spinner.bind(text=self.on_resolution_select)
        layout.add_widget(self.resolution_spinner)

        # Button to enhance image
        self.enhance_button = Button(text="Enhance Image", size_hint_y=None, height=50)
        self.enhance_button.bind(on_press=self.enhance_image)
        layout.add_widget(self.enhance_button)

        # Button to generate caption
        self.caption_button = Button(text="Generate Caption", size_hint_y=None, height=50)
        self.caption_button.bind(on_press=self.generate_caption)
        layout.add_widget(self.caption_button)

        # Label to display caption
        self.caption_label = Label(text="Generated Caption: ", size_hint_y=None, height=50)
        layout.add_widget(self.caption_label)

        # Image display layout
        self.image_display_layout = BoxLayout(orientation='horizontal')
        layout.add_widget(self.image_display_layout)

        # Original image widget
        self.original_image_widget = KivyImage(size_hint=(1, 1))
        self.image_display_layout.add_widget(self.original_image_widget)

        # Enhanced image widget
        self.enhanced_image_widget = KivyImage(size_hint=(1, 1))
        self.image_display_layout.add_widget(self.enhanced_image_widget)

        # Create a horizontal layout for the labels
        label_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=40)

        # Labels for original and enhanced images
        self.original_image_label = Label(text="Original Image", size_hint_x=None, width=500)
        self.enhanced_image_label = Label(text="Enhanced Image", size_hint_x=None, width=500)

        # Add labels to the horizontal layout
        label_layout.add_widget(self.original_image_label)
        label_layout.add_widget(self.enhanced_image_label)
        
        # Add the label layout to the main layout
        layout.add_widget(label_layout)

        return layout

    def upload_image(self, instance):
        """Open the file chooser in a popup to select an image file."""
        content = CustomFileChooser()
        content.bind(on_submit=self.load_image_from_chooser)
        
        # Create a Popup to show the file chooser
        self.popup = Popup(title="Select an Image", content=content, size_hint=(0.9, 0.9))
        self.popup.open()

    def load_image_from_chooser(self, chooser, selection, touch):
        """Load image path from file chooser."""
        if selection:
            selected_file = selection[0]
            if os.path.isfile(selected_file) and selected_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                self.image_path_input.text = selected_file
                self.popup.dismiss()  # Close the popup after selection

    def on_resolution_select(self, spinner, text):
        """Update the selected resolution when spinner value changes."""
        self.selected_resolution = text

    def enhance_image(self, instance):
        image_path = self.image_path_input.text

        if not os.path.exists(image_path):
            print("Image file not found.")
            return

        # Load the image
        img = cv2.imread(image_path)
        if img is None:
            print("Could not load image.")
            return

        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Add a slight blur to the original image for comparison
        original_img_blurred = cv2.GaussianBlur(img_rgb, (3, 3), 0)
        original_img_path = os.path.join(os.path.dirname(image_path), "original_blurred_" + os.path.basename(image_path))
        cv2.imwrite(original_img_path, cv2.cvtColor(original_img_blurred, cv2.COLOR_RGB2BGR))

        # Apply enhancement based on selected resolution
        if not hasattr(self, 'selected_resolution'):
            print("Please select a resolution.")
            return

        # Open image in PIL for contrast, sharpness, and detail adjustments
        pil_img = Image.fromarray(img_rgb)

        if self.selected_resolution == 'low':
            enhanced_img = self.apply_enhancements(pil_img, contrast=1.05, sharpness=1.05, clarity=1.1)
        elif self.selected_resolution == 'medium':
            enhanced_img = self.apply_enhancements(pil_img, contrast=1.15, sharpness=1.15, clarity=1.2)
        elif self.selected_resolution == 'high':
            enhanced_img = self.apply_enhancements(pil_img, contrast=1.22, sharpness=1.2, clarity=1.35)

        # Save enhanced image
        enhanced_img_path = os.path.join(os.path.dirname(image_path), "enhanced_" + os.path.basename(image_path))
        enhanced_img.save(enhanced_img_path)

        # Update the image display widgets
        self.original_image_widget.texture = self.load_image(original_img_path)
        self.enhanced_image_widget.texture = self.load_image(enhanced_img_path)

    def apply_enhancements(self, pil_img, contrast, sharpness, clarity):
        """Apply contrast, sharpness, and clarity enhancements."""
        enhancer_contrast = ImageEnhance.Contrast(pil_img)
        enhanced_img = enhancer_contrast.enhance(contrast)

        enhancer_sharpness = ImageEnhance.Sharpness(enhanced_img)
        enhanced_img = enhancer_sharpness.enhance(sharpness)

        enhancer_clarity = ImageEnhance.Brightness(enhanced_img)
        enhanced_img = enhancer_clarity.enhance(clarity)

        return enhanced_img

    def load_image(self, image_path):
        """Load an image and convert it to a texture for Kivy.""" 
        img = Image.open(image_path)

        # Resize to fit widget, converting dimensions to integers
        img = img.resize((int(self.original_image_widget.width), int(self.original_image_widget.height)), Image.LANCZOS)
        
        # Convert image to Kivy texture
        img_texture = self.image_to_texture(img)
        return img_texture

    def image_to_texture(self, img):
        """Convert a PIL image to a Kivy texture."""
        img_data = img.tobytes()
        texture = Texture.create(size=img.size)
        texture.blit_buffer(img_data, colorfmt='rgb', bufferfmt='ubyte')
        texture.flip_vertical()
        return texture

    def load_model_if_not_loaded(self):
        """Lazy load the model when it's first needed."""
        global caption_model, feature_extractor, tokenizer
        if caption_model is None:
            print("Loading caption model...")
            caption_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning").to(device)
            feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
            tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
            print("Model loaded successfully.")

    def generate_caption(self, instance):
        """Generate a caption for the currently enhanced image."""
        # Lazy load the captioning model
        self.load_model_if_not_loaded()

        # Get the path of the enhanced image
        enhanced_image_path = os.path.join(os.path.dirname(self.image_path_input.text), 
                                           "enhanced_" + os.path.basename(self.image_path_input.text))

        if not os.path.exists(enhanced_image_path):
            print("Enhanced image not found.")
            return

        # Load the enhanced image
        img = Image.open(enhanced_image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Prepare the image for the captioning model
        pixel_values = feature_extractor(images=img, return_tensors="pt").pixel_values.to(device)

        # Generate caption
        outputs = caption_model.generate(pixel_values, max_length=16, num_beams=4)
        caption = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Display caption
        self.caption_label.text = f"Generated Caption: {caption}"

if __name__ == '__main__':
    ImageCaptionApp().run()
