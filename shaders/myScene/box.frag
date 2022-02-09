#version 450

layout (location = 0) in vec3 inNormal;
layout (location = 1) in vec3 inColor;
layout (location = 2) in vec2 inUV;
layout (location = 3) in vec3 inViewVec;
layout (location = 4) in vec3 inLightVec;

layout(set = 0, binding = 1) uniform SELECT {
    float map;
} select;

layout(set = 0, binding = 2) uniform INFO {
    float map;
} info;


layout (location = 0) out vec4 outFragColor;

void main()
{

    if (select.map == 0){
    }

    if (select.map == 1){
    }
    if (select.map == 2){

    }


    outFragColor = vec4(1.0f, 1.0f, 1.0f, 1.0f);

    //color = vec4(0.3, 0.3, 0.3, 1.0);
    //outFragColor = color;

}