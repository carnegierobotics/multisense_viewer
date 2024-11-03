#version 450

layout (location = 0) in vec2 inUV;
layout (binding = 0) uniform sampler2D samplerColorMap;
layout (location = 0) out vec4 outColor;

layout (binding = 1) uniform INFO {
    int selection;
} info;

// Function to apply Jet colormap
vec3 jetColormap(float value) {
    value = clamp(value, 0.0, 1.0);
    float r = smoothstep(0.375, 0.625, value) * (value >= 0.5 ? 1.0 : 0.0);
    float g = smoothstep(0.125, 0.875, value);
    float b = smoothstep(0.375, 0.625, 1.0 - value) * (value < 0.5 ? 1.0 : 0.0);
    return vec3(r, g, b);
}

// Function to apply Viridis colormap
vec3 viridisColormap(float value) {
    value = clamp(value, 0.0, 1.0);
    float r = smoothstep(0.0, 1.0, value) * (value < 0.75 ? 1.0 : 0.0);
    float g = smoothstep(0.25, 0.75, value);
    float b = smoothstep(0.5, 1.0, value) * (value >= 0.25 ? 1.0 : 0.0);
    return vec3(r, g, b);
}

void main() {
    float color = texture(samplerColorMap, inUV).r;
    vec3 finalColor;
    if (info.selection == 0) { // DepthColorOption::None
        finalColor = texture(samplerColorMap, inUV).rgb;
    }
    else if (info.selection == 1) { // DepthColorOption::Invert
        finalColor = vec3(1.0 - color);
    }
    else if (info.selection == 2) { // DepthColorOption::Normalize
        finalColor = vec3(color);
    }

    else {
        finalColor = vec3(color); // Default case, just in case of an invalid selection
    }

    outColor = vec4(finalColor, 1.0);
}
