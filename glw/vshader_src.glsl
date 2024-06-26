// #version 330 core

// in vec4 a_Position;
// in vec4 a_Color;
// in vec4 a_Normal;



// out vec4 v_Color;
// uniform mat4 u_ProjMatrix;
// uniform mat4 u_ViewMatrix;
// uniform mat4 u_ModelMatrix;

// void main() { 
//     gl_Position = u_ProjMatrix * u_ViewMatrix * u_ModelMatrix * a_Position; 
//     v_Color = a_Color;
// }
#version 330 core

in vec4 a_Position;
in vec3 a_Normal;
in vec4 a_Color;
in vec2 a_Texcoord;

uniform mat4 u_ProjMatrix;
uniform mat4 u_ViewMatrix;
uniform mat4 u_ModelMatrix;

uniform vec3 u_CamPos;

out vec2 v_Texcoord;
out vec3 v_Normal;
out vec3 v_CamDir;
out vec4 v_Color;

flat out int simpleRender;

void main() { 
    gl_Position = u_ProjMatrix * u_ViewMatrix * u_ModelMatrix * a_Position; 
    v_Texcoord = a_Texcoord;
    v_Color = a_Color;

    
    if (a_Normal == vec3(0.0, 0.0, 0.0)) {
        simpleRender = 1;
    }
    else {
        simpleRender = 0;
        mat4 NormalMatrix = transpose(inverse(u_ModelMatrix)); // 法向量矩阵
        v_Normal = normalize(vec3(NormalMatrix * vec4(a_Normal, 1.0))); // 重新计算模型变换后的法向量
        v_CamDir = normalize(u_CamPos - vec3(u_ModelMatrix * a_Position)); // 从当前顶点指向相机的向量

    }

    // mat4 NormalMatrix = transpose(inverse(u_ModelMatrix)); // 法向量矩阵
    // v_Normal = normalize(vec3(NormalMatrix * vec4(a_Normal, 1.0))); // 重新计算模型变换后的法向量
    // v_CamDir = normalize(u_CamPos - vec3(u_ModelMatrix * a_Position)); // 从当前顶点指向相机的向量
}
