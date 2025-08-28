#version 330 core

layout (location=0) in vec4 a_Position;
layout (location=1) in vec4 a_Color;
layout (location=2) in vec3 a_Normal;
layout (location=3) in vec2 a_Texcoord;

struct PointLight {
    vec3 position;
    vec3 color;
};

uniform mat4 u_ProjMatrix;
uniform mat4 u_ViewMatrix;
uniform mat4 u_ModelMatrix;

uniform vec3 u_CamPos;
uniform int u_farPlane;
uniform float u_farPlaneRatio;
uniform float u_pointSize;

out vec3 _v_Position;
out vec3 _v_Normal;
out vec4 _v_Color;
out vec2 _v_Texcoord;
out vec3 _v_WorldSpaceCamPos;
flat out int _v_simpleRender;

void main() {
    gl_Position = u_ProjMatrix * u_ViewMatrix * u_ModelMatrix * a_Position;
    
    _v_Color = a_Color;
    _v_Texcoord = a_Texcoord;
    _v_WorldSpaceCamPos = u_CamPos; 

    gl_PointSize = u_pointSize;

    if (a_Normal == vec3(0.0, 0.0, 0.0)) {
        _v_simpleRender = 1;

    }
    else {
        _v_simpleRender = 0;

        _v_Position = vec3(u_ModelMatrix * a_Position);
        _v_Normal = mat3(transpose(inverse(u_ModelMatrix))) * a_Normal;
    }
    
    if (u_farPlane == 1) {
        _v_simpleRender = 1;
        vec3 vertex_distance = vec3(u_ModelMatrix * a_Position) - u_CamPos;
        float distance_factor = 1.0 - clamp(length(vertex_distance) * u_farPlaneRatio, 0.0, 1.0);

        vec3 worldPos = vec3(u_ModelMatrix * a_Position);
        vec3 viewDir = normalize(u_CamPos - vec3(0.0, 0.0, 0.0));

        float viewNormalDot = 0.0;


        if (a_Normal == vec3(0.0, 0.0, 0.0)) {
            vec3 normal = vec3(0.0, 0.0, 1.0);
            viewNormalDot = clamp(abs(dot(normal, viewDir)) * 4, 0.0, 1.0);
        }
        else {
            vec4 word_normal_4 = vec4(u_ModelMatrix * vec4(a_Normal, 1.0));
            vec3 word_normal = normalize(word_normal_4.xyz);
            viewNormalDot = clamp(abs(dot(word_normal, viewDir)) * 4, 0.0, 1.0);
        }

        
        _v_Color = a_Color * distance_factor * viewNormalDot;
    }

}