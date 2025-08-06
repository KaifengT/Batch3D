#version 330 core

struct PointLight {
    vec3 position;
    vec3 color;
};

#define MAX_LIGHTS 10

uniform PointLight u_Lights[MAX_LIGHTS];
uniform int u_NumLights;

in vec3 v_Position;
in vec3 v_Normal;
in vec4 v_Color;
in vec2 v_Texcoord;
in vec3 v_WorldSpaceCamPos;
flat in int simpleRender;

uniform sampler2D u_Texture;

uniform vec3 u_AmbientColor; // 环境光颜色
uniform float u_Shiny; // 高光系数，非负数，数值越大高光点越小
uniform float u_Specular; // 镜面反射系数，0~1之间的浮点数，影响高光亮度
uniform float u_Diffuse; // 漫反射系数，0~1之间的浮点数，影响表面亮度
uniform float u_Pellucid; // 透光系数，0~1之间的浮点数，影响背面亮度

uniform int render_mode;


void main() {

    if (simpleRender == 0){

        vec3 normal = normalize(v_Normal);
        vec3 result = vec3(0.0);

        vec3 viewDir = normalize(v_WorldSpaceCamPos - v_Position);
        vec3 ambient_lighting = u_AmbientColor;
        vec3 diffuse_lighting = vec3(0.0);
        vec3 specular_lighting = vec3(0.0);

        for (int i = 0; i < u_NumLights; ++i) {
            vec3 lightDir = normalize(u_Lights[i].position - v_Position);
            float diff_intensity = max(dot(normal, lightDir), 0.0);
            diffuse_lighting += min(u_Lights[i].color * diff_intensity * u_Diffuse, vec3(1.0));
            // vec3 scatteredLight = min(u_AmbientColor + u_Lights[i].color * diff_intensity, vec3(1.0)); // 散射光

            // result += scatteredLight;
            // diffuse_lighting += u_AmbientColor * u_Lights[i].color * diff_intensity;

            // (Blinn-Phong)
            if (diff_intensity > 0.0) { 
                vec3 halfwayDir = normalize(lightDir + viewDir);
                float spec_angle = max(dot(normal, halfwayDir), 0.0);
                float spec_intensity = pow(spec_angle, u_Shiny);
                specular_lighting += u_Specular * spec_intensity * u_Lights[i].color;
            }
            
        }
        result = ambient_lighting + diffuse_lighting + specular_lighting + u_AmbientColor;

        // render mode texture
        if (render_mode == 3){
            vec4 color = texture2D(u_Texture, v_Texcoord);
            vec3 rgb = min(color.rgb * result, vec3(1.0));
            gl_FragColor = vec4(rgb, 1.0);
            return;
        }
        // render mode normal
        else if (render_mode == 2){
            
            
            gl_FragColor = vec4((1.0-normal)*0.4 + 0.2, 1.0);
            return;
        }
        // render mode none
        else{
            vec3 rgb = min(v_Color.rgb * result, vec3(1.0));
            gl_FragColor = vec4(rgb, v_Color.a);
            return;
        }

    }

    else{
        gl_FragColor = v_Color;
        return;

    }

}