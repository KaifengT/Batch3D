#version 460 core

layout (location=0) in vec4 a_Position;
layout (location=1) in vec4 a_Color;
layout (location=2) in vec3 a_Normal;
layout (location=3) in vec2 a_Texcoord;

struct PointLight {
    vec3 position;
    vec3 color;
};


uniform mat4 u_ModelMatrix;
uniform mat4 u_mvpMatrix;
uniform mat3 u_worldNormalMatrix;

uniform vec3 u_CamPos;
uniform int u_farPlane;
uniform float u_farPlaneRatio;
uniform float u_pointSize;

out vec3 v_Position;
out vec3 v_Normal;
out vec4 v_Color;
out vec2 v_Texcoord;
out vec3 v_WorldSpaceCamPos;
flat out int v_simpleRender;

void main() {
    gl_Position = u_mvpMatrix * a_Position;

    vec3 worldPos = vec3(u_ModelMatrix * a_Position);

    v_Color = a_Color;
    v_Texcoord = a_Texcoord;
    v_WorldSpaceCamPos = u_CamPos; 

    gl_PointSize = u_pointSize;

    if (a_Normal == vec3(0.0, 0.0, 0.0)) {
        v_simpleRender = 1;

    }
    else {
        v_simpleRender = 0;

        v_Position = worldPos;
        v_Normal = u_worldNormalMatrix * a_Normal;
    }
    
    if (u_farPlane == 1) {
        v_simpleRender = 1;
        vec3 vertex_distance = worldPos - u_CamPos;
        float distance_factor = 1.0 - clamp(length(vertex_distance) * u_farPlaneRatio, 0.0, 1.0);

        vec3 viewDir = normalize(u_CamPos - vec3(0.0, 0.0, 0.0));

        float viewNormalDot = 0.0;


        if (a_Normal == vec3(0.0, 0.0, 0.0)) {
            vec3 normal = vec3(0.0, 0.0, 1.0);
            viewNormalDot = clamp(abs(dot(normal, viewDir)) * 4, 0.0, 1.0);
        }
        else {
            vec4 world_normal_4 = vec4(u_ModelMatrix * vec4(a_Normal, 1.0));
            vec3 world_normal = normalize(world_normal_4.xyz);
            viewNormalDot = clamp(abs(dot(world_normal, viewDir)) * 4, 0.0, 1.0);
        }

        
        v_Color = a_Color * distance_factor * viewNormalDot;
    }

}