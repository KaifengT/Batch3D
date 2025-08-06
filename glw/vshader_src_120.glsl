#version 120

attribute vec4 a_Position;
attribute vec3 a_Normal;
attribute vec4 a_Color;
attribute vec2 a_Texcoord;

struct PointLight {
    vec3 position;
    vec3 color;
};

uniform mat4 u_ProjMatrix;
uniform mat4 u_ViewMatrix;
uniform mat4 u_ModelMatrix;

uniform vec3 u_CamPos;
uniform int u_farPlane;
uniform float u_farPlane_ratio;

varying vec3 v_Position;
varying vec3 v_Normal;
varying vec4 v_Color;
varying vec2 v_Texcoord;
varying vec3 v_WorldSpaceCamPos;
varying float simpleRender;

void main() {
    gl_Position = u_ProjMatrix * u_ViewMatrix * u_ModelMatrix * a_Position;
    
    v_Color = a_Color;
    v_Texcoord = a_Texcoord;
    v_WorldSpaceCamPos = u_CamPos; 

    if (u_farPlane == 1) {
        simpleRender = 1.0;
        vec3 vertex_distance = vec3(u_ModelMatrix * a_Position) - u_CamPos;
        float distance_factor = 1.0 - clamp(length(vertex_distance) * u_farPlane_ratio, 0.0, 1.0);

        vec3 worldPos = vec3(u_ModelMatrix * a_Position);
        vec3 viewDir = normalize(u_CamPos - vec3(0.0, 0.0, 0.0));

        float viewNormalDot = 0.0;

        if (a_Normal == vec3(0.0, 0.0, 0.0)) {
            vec3 normal = vec3(0.0, 0.0, 1.0);
            viewNormalDot = clamp(abs(dot(normal, viewDir)) * 4.0, 0.0, 1.0);
        }
        else {
            vec4 word_normal_4 = vec4(u_ModelMatrix * vec4(a_Normal, 1.0));
            vec3 word_normal = normalize(word_normal_4.xyz);
            viewNormalDot = clamp(abs(dot(word_normal, viewDir)) * 4.0, 0.0, 1.0);
        }

        v_Color = a_Color * distance_factor * viewNormalDot;
    }

    if (a_Normal == vec3(0.0, 0.0, 0.0)) {
        simpleRender = 1.0;
    }
    else {
        simpleRender = 0.0;

        v_Position = vec3(u_ModelMatrix * a_Position);
        // v_Normal = mat3(transpose(inverse(u_ModelMatrix))) * a_Normal;
        v_Normal = normalize(vec3(u_ModelMatrix * vec4(a_Normal, 0.0)));
    }
}