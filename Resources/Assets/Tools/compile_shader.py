import os
import shutil
import subprocess
import sys
import glob
import platform

def compile_shader(glslc, source, destination):
    subprocess.run([glslc, source, '-o', destination], check=True)

def should_recompile(source, destination):
    if not os.path.exists(destination):
        return True
    return os.path.getmtime(source) > os.path.getmtime(destination)
def main():
    if platform.system() == "Windows":
        is_windows = True
    elif platform.system() == "Linux":
        is_windows = False
    else:
        is_windows = None  # For other systems
    project_path = os.path.abspath(os.path.join(os.getcwd()))

    if is_windows:
        print(f"Compiling from Windows: cwd: {os.getcwd()}")
        print(f"Compiling from Windows: project path: {project_path}")
        glslc = "Windows/glslc.exe"
    else:
        print(f"Compiling from Ubuntu: cwd: {os.getcwd()}")
        glslc = "/usr/bin/glslc"

    scene_out_dir = os.path.join(project_path, "Resources/Assets/Shaders/spv/")
    shader_dir = os.path.join(project_path, "Resources/Assets/Shaders")

    print(f"glslc path: {glslc}")
    print(f"Scene output directory: {scene_out_dir}")
    print(f"Shader directory: {shader_dir}")

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

    # Find all cmake-build-* directories
    build_dirs = [d for d in glob.glob(os.path.join(project_path, "cmake-build-*")) if os.path.isdir(d)]

    # Copy compiled shaders to each build directory
    for build_dir in build_dirs:
        out_dir = os.path.join(build_dir, "Resources/Assets/Shaders/spv/")
        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        print(f"Copying to build location: {scene_out_dir}*.spv | to | {out_dir}")
        for file in glob.glob(os.path.join(scene_out_dir, '**', '*.spv'), recursive=True):
            try:
                if not os.path.exists(os.path.join(out_dir, os.path.relpath(file, scene_out_dir))):
                    os.makedirs(os.path.join(out_dir, os.path.relpath(file, scene_out_dir)), exist_ok=True)
                shutil.copy(file, os.path.join(out_dir, os.path.relpath(file, scene_out_dir)))
            except Exception as e:
                print(e)
    print("Exiting...")

if __name__ == "__main__":
    main()