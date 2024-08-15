import os
import shutil
import subprocess
import sys
import glob

def compile_shader(glslc, source, destination):
    subprocess.run([glslc, source, '-o', destination], check=True)

def should_recompile(source, destination):
    if not os.path.exists(destination):
        return True
    return os.path.getmtime(source) > os.path.getmtime(destination)

def main():
    is_windows = len(sys.argv) > 1 and sys.argv[1] == "windows"
    project_path = os.path.abspath(os.path.join(os.getcwd(), "../../.."))

    if is_windows:
        print(f"Compiling from Windows: cwd: {os.getcwd()}")
        print(f"Compiling from Windows: project path: {project_path}")

        glslc = "glslc.exe"
        scene_out_dir = os.path.join(project_path, "Assets/Shaders/spv/")
        out_dir = os.path.join(project_path, "cmake-build-debug/Assets/Shaders/spv/")
        shader_dir = os.path.join(project_path, "Assets/Shaders")
    else:
        print(f"Compiling from ubuntu {os.getcwd()}")
        glslc = "/usr/bin/glslc"
        out_dir = os.path.abspath(os.path.join(os.getcwd(), "../../cmake-build-debug/Assets/Shaders/spv/"))
        scene_out_dir = os.path.abspath(os.path.join(os.getcwd(), "../Shaders/spv/"))
        shader_dir = os.path.abspath(os.path.join(os.getcwd(), "../Shaders"))

    os.makedirs(scene_out_dir, exist_ok=True)

    # Use glob to find all shader files in the shader directory and its subdirectories
    shader_files = glob.glob(os.path.join(shader_dir, '**', '*.frag'), recursive=True) + \
                   glob.glob(os.path.join(shader_dir, '**', '*.vert'), recursive=True)

    recompiled_files = []

    for source_file in shader_files:
        # Determine the output file name by replacing the shader directory path and extension
        output_file = os.path.relpath(source_file, shader_dir).replace('.frag', '.frag.spv').replace('.vert', '.vert.spv')
        output_path = os.path.join(scene_out_dir, output_file)

        # Ensure the directory for the output file exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Check if the shader should be recompiled
        if should_recompile(source_file, output_path):
            compile_shader(glslc, source_file, output_path)
            recompiled_files.append(source_file)

    if recompiled_files:
        print("Recompiled shaders:")
        for file in recompiled_files:
            print(f"- {file}")
    else:
        print("No shaders needed recompiling.")

    print(f"Copying to debug build location: {scene_out_dir}*.spv | to | {out_dir}")
    for file in glob.glob(os.path.join(scene_out_dir, '**', '*.spv'), recursive=True):
        shutil.copy(file, out_dir)

    if is_windows:
        input("Press any key to exit...")

    print("Exiting...")

if __name__ == "__main__":
    main()
