#version 450
layout(location = 0) in float near; //0.01
layout(location = 1) in float far; //100
layout(location = 2) in vec3 nearPoint;
layout(location = 3) in vec3 farPoint;
layout(location = 4) in mat4 fragProj;
layout(location = 8) in mat4 fragView;
layout(location = 0) out vec4 outColor;


vec4 grid(vec3 fragPos3D, float scale) {
    vec2 coord = fragPos3D.xz * scale;
    vec2 derivative = fwidth(coord);

    vec2 grid = abs(fract(coord) - 0.5) / derivative;
    float intensity = 1.0 - min(grid.x, grid.y); // Closer to grid lines, intensity is higher

    float minimumz = min(derivative.y, 1);
    float minimumx = min(derivative.x, 1);
    vec4 color = vec4(vec3(intensity) * 0.3, 1.0);

    if(fragPos3D.x > -0.1 * minimumx && fragPos3D.x < 0.1 * minimumx)
    color.z = 1.0;

    if(fragPos3D.z > -0.1 * minimumz && fragPos3D.z < 0.1 * minimumz)
    color.x = 1.0;

    return color;
}

float computeLinearDepth(vec3 pos) {
    vec4 clip_space_pos = fragProj * fragView * vec4(pos.xyz, 1.0);
    float clip_space_depth = (clip_space_pos.z / clip_space_pos.w) * 2.0 - 1.0; // put back between -1 and 1
    float linearDepth = (2.0 * near * far) / (far + near - clip_space_depth * (far - near)); // get linear value between 0.01 and 100
    return linearDepth / far; // normalize
}

void main() {
    float t = -nearPoint.y / (farPoint.y - nearPoint.y);
    vec3 fragPos3D = nearPoint + t * (farPoint - nearPoint);

    outColor = grid(fragPos3D, 10) * float(t > 0);;

    float linearDepth = computeLinearDepth(fragPos3D);
    float maxFadeDistance = 0.1; // Adjust as needed
    float fading = clamp((maxFadeDistance - linearDepth) / maxFadeDistance, 0.0, 1.0);

    outColor.rgb *= fading;
}