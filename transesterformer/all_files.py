import os

def dump_all_files_with_contents(root_dir, output_file):
    with open(output_file, 'w', encoding='utf-8') as out:
        for current_dir, _, files in os.walk(root_dir):
            for file in files:
                file_path = os.path.join(current_dir, file)
                rel_path = os.path.relpath(file_path, root_dir)
                
                out.write(f"\n\n========== FILE: {rel_path} ==========\n")
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        contents = f.read()
                        out.write(contents)
                except Exception as e:
                    out.write(f"[ERROR READING FILE: {e}]")

if __name__ == "__main__":
    current_folder = os.path.dirname(os.path.abspath(__file__))
    output_txt = os.path.join(current_folder, "all_files_with_contents.txt")

    dump_all_files_with_contents(current_folder, output_txt)
    print(f"✅ Dump complete. Output saved to: {output_txt}")
