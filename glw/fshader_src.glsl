// #version 330 core

// in vec4 v_Color;


// void main() { 
//     gl_FragColor = v_Color;  // gl_FragColor是内置变量
// } 


#version 330 core

struct PointLight {
    vec3 position;
    vec3 color;
};

#define MAX_LIGHTS 10

uniform PointLight u_Lights[MAX_LIGHTS];
uniform int u_NumLights;

in vec2 v_Texcoord;
in vec3 v_Normal;
in vec3 v_CamDir;
in vec4 v_Color;
flat in int simpleRender;

uniform sampler2D u_Texture;
uniform vec3 u_LightDir; // 定向光方向
uniform vec3 u_LightColor; // 定向光颜色
uniform vec3 u_AmbientColor; // 环境光颜色
uniform float u_Shiny; // 高光系数，非负数，数值越大高光点越小
uniform float u_Specular; // 镜面反射系数，0~1之间的浮点数，影响高光亮度
uniform float u_Diffuse; // 漫反射系数，0~1之间的浮点数，影响表面亮度
uniform float u_Pellucid; // 透光系数，0~1之间的浮点数，影响背面亮度



void main() {


    if (simpleRender == 0) {

        vec3 lightDir = normalize(-u_LightDir); // 光线向量取反后单位化
        vec3 middleDir = normalize(v_CamDir + lightDir); // 视线和光线的中间向量
        // vec4 color = texture2D(u_Texture, v_Texcoord);
        vec4 color = v_Color;
        

        float diffuseCos = u_Diffuse * max(0.0, dot(lightDir, v_Normal)); // 光线向量和法向量的内积
        float specularCos = u_Specular * max(0.0, dot(middleDir, v_Normal)); // 中间向量和法向量内积

        if (!gl_FrontFacing) 
            diffuseCos *= u_Pellucid; // 背面受透光系数影响

        if (diffuseCos == 0.0) 
            specularCos = 0.0;
        else
            specularCos = pow(specularCos, u_Shiny);

        vec3 scatteredLight = min(u_AmbientColor + u_LightColor * diffuseCos, vec3(1.0)); // 散射光
        vec3 reflectedLight = u_LightColor * specularCos; // 反射光
        vec3 rgb = min(color.rgb * (scatteredLight + reflectedLight), vec3(1.0));


    
        gl_FragColor = vec4(rgb, color.a);
        return;
    }
    else {
        gl_FragColor = v_Color;
        return;
    }
    // gl_FragColor = vec4(rgb, color.a);
    // gl_FragColor = v_Color;
} 

