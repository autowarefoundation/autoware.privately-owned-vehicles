# © 2025 TranHuuNhatHuy <huy9515@gmail.com>
# Autoware POV Frame-by-frame Manual Audit tool

import os
import json
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk


STATE_FILE = "audit_state.json"
STATUS_FLASH_MS = 200
FLASH_BORDER_W = 10


class ImageAuditorApp:
    def __init__(self, root):

        self.root = root
        self.root.title("Antoware POV - Tran's Frame Manual Auditor")

        self.border_frame = tk.Frame(root, bd = FLASH_BORDER_W)
        self.border_frame.pack()
        self.image_label = tk.Label(self.border_frame)
        self.image_label.pack()

        self.accepted_images = []
        self.rejected_images = []
        self.image_files = []
        self.current_index = 0

        self.select_folder_and_resume()
        self.setup_bindings()
        self.setup_buttons_counters()
        self.update_counters()


    def setup_bindings(self):
        self.root.bind("1", self.accept_image)
        self.root.bind("2", self.reject_image)
        self.root.bind("3", self.end_session)

    
    def update_counters(self):
        self.accepted_var.set(f"Accepted: {len(self.accepted_images)}")
        self.rejected_var.set(f"Rejected: {len(self.rejected_images)}")
        self.total_var.set(f"Total: {len(self.accepted_images) + len(self.rejected_images)}")


    def setup_buttons_counters(self):
        # For buttons
        frame = tk.Frame(self.root)
        frame.pack(fill = tk.X)

        list_btns = [
            # Accept
            tk.Button(
                frame, 
                text = "1: Accept", 
                command = self.accept_image
            ),
            # Reject
            tk.Button(
                frame, 
                text = "2: Reject", 
                command = self.reject_image
            ),
            # Save & Quit
            tk.Button(
                frame, 
                text = "3: Save & quit", 
                command = self.end_session
            )
        ]

        for btn in list_btns:
            btn.pack(
                side = tk.LEFT, 
                expand = True, 
                fill = tk.X
            )

        # For counter textboxes
        counter_frame = tk.Frame(self.root)
        counter_frame.pack(fill = tk.X)

        self.accepted_var = tk.StringVar()
        self.rejected_var = tk.StringVar()
        self.total_var = tk.StringVar()

        self.update_counters()

        for var in [self.accepted_var, self.rejected_var, self.total_var]:
            tk.Label(
                counter_frame, 
                textvariable = var, 
                relief = "sunken"
            ).pack(
                side = tk.LEFT, 
                expand = True, 
                fill = tk.X
            )


    def select_folder_and_resume(self):
        folder_path = filedialog.askdirectory(
            title = "Select folder containing PNG images"
        )
        if not folder_path:
            messagebox.showerror("Error", "No folder selected. Exiting.")
            self.root.quit()
            return

        self.image_files = sorted([
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith(".png")
        ])

        if not self.image_files:
            messagebox.showinfo("Info", "No PNG images found. Try again lol.")
            self.root.quit()
            return

        # Try to resume
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, "r") as f:
                state = json.load(f)
                if state.get("folder") == folder_path:
                    self.current_index = state.get("index", 0)
                    self.accepted_images = state.get("accepted", [])
                    self.rejected_images = state.get("rejected", [])

        self.show_image()


    def show_image(self):
        self.border_frame.config(highlightthickness = 0)

        if self.current_index >= len(self.image_files):
            self.finish_review()
            return

        img_path = self.image_files[self.current_index]
        img = Image.open(img_path)
        img.thumbnail((800, 800))
        img_tk = ImageTk.PhotoImage(img)

        self.image_label.configure(image = img_tk)
        self.image_label.image = img_tk
        self.root.title(f"[{self.current_index+1}/{len(self.image_files)}] Reviewing: {os.path.basename(img_path)}")


    def accept_image(self, event = None):
        self.accepted_images.append(self.image_files[self.current_index])
        self.update_counters()
        self.flash_feedback("accepted")
        self.current_index += 1
        self.save_state()
        self.root.after(STATUS_FLASH_MS, self.show_image)


    def reject_image(self, event = None):
        self.rejected_images.append(self.image_files[self.current_index])
        self.update_counters()
        self.flash_feedback("rejected")
        self.current_index += 1
        self.save_state()
        self.root.after(STATUS_FLASH_MS, self.show_image)

    
    def clear_border_and_continue(self):
        self.border_frame.config(highlightthickness = 0)


    def flash_feedback(self, status):
        color = (
            "green" if (status == "accepted")
            else "red"
        )
        self.border_frame.config(
            highlightbackground = color, 
            highlightcolor = color, 
            highlightthickness = FLASH_BORDER_W
        )


    def end_session(self, event = None):
        self.save_state()

        save_file = filedialog.asksaveasfilename(
            defaultextension = ".json",
            filetypes = [("JSON files", "*.json")],
            title = "Save audit results"
        )

        if save_file:
            results = {
                "accepted": self.accepted_images,
                "rejected": self.rejected_images
            }
            with open(save_file, "w") as f:
                json.dump(
                    results, f, 
                    indent = 4
                )
            messagebox.showinfo("Saved", f"Results saved to:\n{save_file}")

        if (
            os.path.exists(STATE_FILE) and
            self.current_index >= len(self.image_files)
        ):
            os.remove(STATE_FILE)
        self.root.quit()


    def finish_review(self):
        messagebox.showinfo("Done", "All images have been reviewed!")
        self.end_session()


    def save_state(self):
        folder_path = (
            os.path.dirname(self.image_files[0]) 
            if self.image_files else ""
        )
        state = {
            "folder": folder_path,
            "index": self.current_index,
            "accepted": self.accepted_images,
            "rejected": self.rejected_images
        }
        with open(STATE_FILE, "w") as f:
            json.dump(
                state, f,
                indent = 4
            )


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageAuditorApp(root)
    root.mainloop()