import os
import re

def delete_files_from_warnings(warning_text, base_dir):
    """
    Deletes image files and their corresponding text files based on warning messages.

    :param warning_text: String containing the warning messages.
    :param base_dir: Base directory where the files are located.
    """
    # Regular expression to find file paths in the warning messages
    pattern = re.compile(r"'(.*?\.jpg)'")

    # Find all matches in the warning text
    matches = pattern.findall(warning_text)

    for match in matches:
        # Construct full paths for the image and text file
        image_path = os.path.join(base_dir, match)
        text_path = os.path.splitext(image_path)[0] + '.txt'

        # Delete the image file
        if os.path.exists(image_path):
            os.remove(image_path)
            print(f"Deleted {image_path}")

        # Delete the corresponding text file
        if os.path.exists(text_path):
            os.remove(text_path)
            print(f"Deleted {text_path}")
if __name__ == '__main__':
    # Example usage
    warning_text = """warning: in the working copy of 'CC12M/image_27.jpg', LF will be replaced by CRLF the next time Git touches it
    warning: in the working copy of 'CC12M/image_359.jpg', LF will be replaced by CRLF the next time Git touches it
    warning: in the working copy of 'CC12M/image_53.jpg', LF will be replaced by CRLF the next time Git touches it
    warning: in the working copy of 'CC12M/image_56.jpg', LF will be replaced by CRLF the next time Git touches it
    warning: in the working copy of 'CC12M/image_739.jpg', LF will be replaced by CRLF the next time Git touches it
    warning: in the working copy of 'CC12M/image_753.jpg', LF will be replaced by CRLF the next time Git touches it
    warning: in the working copy of 'CC12M/image_76.jpg', LF will be replaced by CRLF the next time Git touches it
    warning: in the working copy of 'CC12M/image_982.jpg', LF will be replaced by CRLF the next time Git touches it
    warning: in the working copy of 'CC12M/image_1019.jpg', LF will be replaced by CRLF the next time Git touches it
    warning: in the working copy of 'CC12M/image_1032.jpg', LF will be replaced by CRLF the next time Git touches it
    warning: in the working copy of 'CC12M/image_1300.jpg', LF will be replaced by CRLF the next time Git touches it
    warning: in the working copy of 'CC12M/image_1326.jpg', LF will be replaced by CRLF the next time Git touches it
    warning: in the working copy of 'CC12M/image_1482.jpg', LF will be replaced by CRLF the next time Git touches it
    warning: in the working copy of 'CC12M/image_1553.jpg', LF will be replaced by CRLF the next time Git touches it
    warning: in the working copy of 'CC12M/image_1740.jpg', LF will be replaced by CRLF the next time Git touches it
    warning: in the working copy of 'CC12M/image_1783.jpg', LF will be replaced by CRLF the next time Git touches it
    warning: in the working copy of 'CC12M/image_1823.jpg', LF will be replaced by CRLF the next time Git touches it
    warning: in the working copy of 'CC12M/image_1836.jpg', LF will be replaced by CRLF the next time Git touches it
    warning: in the working copy of 'CC12M/image_2001.jpg', LF will be replaced by CRLF the next time Git touches it
    warning: in the working copy of 'CC12M/image_2056.jpg', LF will be replaced by CRLF the next time Git touches it
    warning: in the working copy of 'CC12M/image_2066.jpg', LF will be replaced by CRLF the next time Git touches it
    warning: in the working copy of 'CC12M/image_2071.jpg', LF will be replaced by CRLF the next time Git touches it
    warning: in the working copy of 'CC12M/image_2120.jpg', LF will be replaced by CRLF the next time Git touches it
    warning: in the working copy of 'CC12M/image_2391.jpg', LF will be replaced by CRLF the next time Git touches it
    warning: in the working copy of 'CC12M/image_2488.jpg', LF will be replaced by CRLF the next time Git touches it
    warning: in the working copy of 'CC12M/image_2550.jpg', LF will be replaced by CRLF the next time Git touches it
    warning: in the working copy of 'CC12M/image_2954.jpg', LF will be replaced by CRLF the next time Git touches it
    warning: in the working copy of 'CC12M/image_3181.jpg', LF will be replaced by CRLF the next time Git touches it
    warning: in the working copy of 'CC12M/image_3203.jpg', LF will be replaced by CRLF the next time Git touches it
    warning: in the working copy of 'CC12M/image_3249.jpg', LF will be replaced by CRLF the next time Git touches it
    warning: in the working copy of 'CC12M/image_3484.jpg', LF will be replaced by CRLF the next time Git touches it
    warning: in the working copy of 'CC12M/image_3596.jpg', LF will be replaced by CRLF the next time Git touches it
    warning: in the working copy of 'CC12M/image_3606.jpg', LF will be replaced by CRLF the next time Git touches it
    warning: in the working copy of 'CC12M/image_3641.jpg', LF will be replaced by CRLF the next time Git touches it
    warning: in the working copy of 'CC12M/image_3665.jpg', LF will be replaced by CRLF the next time Git touches it
    warning: in the working copy of 'CC12M/image_3666.jpg', LF will be replaced by CRLF the next time Git touches it
    warning: in the working copy of 'CC12M/image_3685.jpg', LF will be replaced by CRLF the next time Git touches it
    warning: in the working copy of 'CC12M/image_3726.jpg', LF will be replaced by CRLF the next time Git touches it
    warning: in the working copy of 'CC12M/image_4010.jpg', LF will be replaced by CRLF the next time Git touches it
    warning: in the working copy of 'CC12M/image_4024.jpg', LF will be replaced by CRLF the next time Git touches it
    warning: in the working copy of 'CC12M/image_4049.jpg', LF will be replaced by CRLF the next time Git touches it
    warning: in the working copy of 'CC12M/image_4173.jpg', LF will be replaced by CRLF the next time Git touches it
    warning: in the working copy of 'CC12M/image_4485.jpg', LF will be replaced by CRLF the next time Git touches it
    warning: in the working copy of 'CC12M/image_4512.jpg', LF will be replaced by CRLF the next time Git touches it
    warning: in the working copy of 'CC12M/image_4519.jpg', LF will be replaced by CRLF the next time Git touches it
    warning: in the working copy of 'CC12M/image_4658.jpg', LF will be replaced by CRLF the next time Git touches it
    warning: in the working copy of 'CC12M/image_4659.jpg', LF will be replaced by CRLF the next time Git touches it
    warning: in the working copy of 'CC12M/image_4846.jpg', LF will be replaced by CRLF the next time Git touches it
    warning: in the working copy of 'CC12M/image_4907.jpg', LF will be replaced by CRLF the next time Git touches it
    warning: in the working copy of 'CC12M/image_4922.jpg', LF will be replaced by CRLF the next time Git touches it
    warning: in the working copy of 'CC12M/image_4952.jpg', LF will be replaced by CRLF the next time Git touches it
    warning: in the working copy of 'CC12M/image_4986.jpg', LF will be replaced by CRLF the next time Git touches it
    warning: in the working copy of 'CC12M/image_5053.jpg', LF will be replaced by CRLF the next time Git touches it
    warning: in the working copy of 'CC12M/image_5246.jpg', LF will be replaced by CRLF the next time Git touches it
    warning: in the working copy of 'CC12M/image_5494.jpg', LF will be replaced by CRLF the next time Git touches it
    warning: in the working copy of 'CC12M/image_5517.jpg', LF will be replaced by CRLF the next time Git touches it
    warning: in the working copy of 'CC12M/image_5719.jpg', LF will be replaced by CRLF the next time Git touches it
    warning: in the working copy of 'CC12M/image_5739.jpg', LF will be replaced by CRLF the next time Git touches it
    warning: in the working copy of 'CC12M/image_5943.jpg', LF will be replaced by CRLF the next time Git touches it
    warning: in the working copy of 'CC12M/image_6051.jpg', LF will be replaced by CRLF the next time Git touches it
    warning: in the working copy of 'CC12M/image_6088.jpg', LF will be replaced by CRLF the next time Git touches it
    warning: in the working copy of 'CC12M/image_6241.jpg', LF will be replaced by CRLF the next time Git touches it
    warning: in the working copy of 'CC12M/image_6334.jpg', LF will be replaced by CRLF the next time Git touches it
    warning: in the working copy of 'CC12M/image_6630.jpg', LF will be replaced by CRLF the next time Git touches it
    warning: in the working copy of 'CC12M/image_6873.jpg', LF will be replaced by CRLF the next time Git touches it
    warning: in the working copy of 'CC12M/image_7198.jpg', LF will be replaced by CRLF the next time Git touches it
    warning: in the working copy of 'CC12M/image_7397.jpg', LF will be replaced by CRLF the next time Git touches it
    warning: in the working copy of 'CC12M/image_7499.jpg', LF will be replaced by CRLF the next time Git touches it
    warning: in the working copy of 'CC12M/image_7552.jpg', LF will be replaced by CRLF the next time Git touches it
    warning: in the working copy of 'CC12M/image_7598.jpg', LF will be replaced by CRLF the next time Git touches it
    warning: in the working copy of 'CC12M/image_7620.jpg', LF will be replaced by CRLF the next time Git touches it
    warning: in the working copy of 'CC12M/image_7903.jpg', LF will be replaced by CRLF the next time Git touches it
    warning: in the working copy of 'CC12M/image_8249.jpg', LF will be replaced by CRLF the next time Git touches it
    warning: in the working copy of 'CC12M/image_8320.jpg', LF will be replaced by CRLF the next time Git touches it
    warning: in the working copy of 'CC12M/image_8355.jpg', LF will be replaced by CRLF the next time Git touches it
    warning: in the working copy of 'CC12M/image_8360.jpg', LF will be replaced by CRLF the next time Git touches it
    warning: in the working copy of 'CC12M/image_8396.jpg', LF will be replaced by CRLF the next time Git touches it
    warning: in the working copy of 'CC12M/image_8659.jpg', LF will be replaced by CRLF the next time Git touches it
    warning: in the working copy of 'CC12M/image_8725.jpg', LF will be replaced by CRLF the next time Git touches it
    warning: in the working copy of 'CC12M/image_8803.jpg', LF will be replaced by CRLF the next time Git touches it
    warning: in the working copy of 'CC12M/image_8836.jpg', LF will be replaced by CRLF the next time Git touches it
    warning: in the working copy of 'CC12M/image_8868.jpg', LF will be replaced by CRLF the next time Git touches it
    warning: in the working copy of 'CC12M/image_9082.jpg', LF will be replaced by CRLF the next time Git touches it
    warning: in the working copy of 'CC12M/image_9085.jpg', LF will be replaced by CRLF the next time Git touches it
    warning: in the working copy of 'CC12M/image_9295.jpg', LF will be replaced by CRLF the next time Git touches it
    warning: in the working copy of 'CC12M/image_9339.jpg', LF will be replaced by CRLF the next time Git touches it
    warning: in the working copy of 'CC12M/image_9635.jpg', LF will be replaced by CRLF the next time Git touches it
    warning: in the working copy of 'CC12M/image_9735.jpg', LF will be replaced by CRLF the next time Git touches it
    warning: in the working copy of 'CC12M/image_9899.jpg', LF will be replaced by CRLF the next time Git touches it
    warning: in the working copy of 'CC12M/image_9983.jpg', LF will be replaced by CRLF the next time Git touches it"""


    base_dir = "CC12M"  # Replace with your base directory path

    delete_files_from_warnings(warning_text, base_dir)
