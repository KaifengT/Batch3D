#version 330 core

in vec4 a_Position;
in vec3 a_Normal;
in vec4 a_Color;
in vec2 a_Texcoord;

struct PointLight {
    vec3 position;
    vec3 color;
};

uniform mat4 u_ProjMatrix;
uniform mat4 u_ViewMatrix;
uniform mat4 u_ModelMatrix;

uniform vec3 u_CamPos;
uniform int u_farPlane;

out vec3 v_Position;
out vec3 v_Normal;
out vec4 v_Color;
out vec2 v_Texcoord;
out vec3 v_WorldSpaceCamPos;
flat out int simpleRender;

void main() {
    gl_Position = u_ProjMatrix * u_ViewMatrix * u_ModelMatrix * a_Position;
    
    v_Color = a_Color;
    v_Texcoord = a_Texcoord;
    v_WorldSpaceCamPos = u_CamPos; 

    if (a_Normal == vec3(0.0, 0.0, 0.0)) {
        simpleRender = 1;

        if (u_farPlane == 1) {
            vec3 vertex_distance = vec3(u_ModelMatrix * a_Position) - u_CamPos;
            v_Color = a_Color - (vec4(0.0, 0.0, 0.0, 1.0) * length(vertex_distance)* 0.02);
        }
    }
    else {
        simpleRender = 0;

        v_Position = vec3(u_ModelMatrix * a_Position);
        v_Normal = mat3(transpose(inverse(u_ModelMatrix))) * a_Normal;
    }
}