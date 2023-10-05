#version 450
layout(location = 0) in float near; //0.01
layout(location = 1) in float far; //100
layout(location = 2) in vec3 nearPoint;
layout(location = 3) in vec3 farPoint;
layout(location = 4) in mat4 fragProj;
layout(location = 8) in mat4 fragView;
layout(location = 0) out vec4 outColor;


vec4 grid(vec3 fragPos3D, float scale) {
    vec2 coord = fragPos3D.xy * scale;
    vec2 derivative = fwidth(coord);

    vec2 grid = abs(fract(coord) - 0.5) / derivative;
    float intensity = 1.0 - min(grid.x, grid.y); // Closer to grid lines, intensity is higher

    float minimumy = min(derivative.y, 1);
    float minimumx = min(derivative.x, 1);

    vec3 baseColor;
    if (intensity > 0.3) { // Close to grid lines
        baseColor = vec3(0.4); // Grid line color
    } else { // Far from grid lines
        baseColor = vec3(0.15); // Black color
    }

    vec4 color = vec4(vec3(intensity) * 0.5, 1.0);
    color = vec4(baseColor, 1.0);

    float axisThickness = 1;
    if(fragPos3D.x > -axisThickness * minimumx && fragPos3D.x < axisThickness * minimumx){
        if (fragPos3D.y > 0)
        color.y = 1.0;
        else
        color.y = 0.5;
    }

    if(fragPos3D.y > -axisThickness * minimumy && fragPos3D.y < axisThickness * minimumy){
        if (fragPos3D.x > 0)
        color.x = 1.0;
        else
        color.x = 0.5;
    }

    return color;
}

float computeLinearDepth(vec3 pos) {
    vec4 clip_space_pos = fragProj * fragView * vec4(pos.xyz, 1.0);
    float clip_space_depth = (clip_space_pos.z / clip_space_pos.w) * 2.0 - 1.0; // put back between -1 and 1
    float linearDepth = (2.0 * near * far) / (far + near - clip_space_depth * (far - near)); // get linear value between 0.01 and 100
    return linearDepth / far; // normalize
}

float computeDepth(vec3 pos) {
    vec4 clip_space_pos = fragProj * fragView * vec4(pos.xyz, 1.0);
    return (clip_space_pos.z / clip_space_pos.w);
}

void main() {
    float t = -nearPoint.z / (farPoint.z - nearPoint.z);
    vec3 fragPos3D = nearPoint + t * (farPoint - nearPoint);

    outColor = grid(fragPos3D, 2) * float(t > 0);;


    float linearDepth = computeLinearDepth(fragPos3D);
    float maxFadeDistance = 0.1; // Adjust as needed
    float fading = clamp((maxFadeDistance - linearDepth) / maxFadeDistance, 0.0, 1.0);

    outColor.rgba *= fading;
}