
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
flat in int v_simpleRender;


uniform int u_renderMode;
uniform int u_FlatShading;

uniform vec2 u_screenSize;
uniform int u_enableAO;

layout(location = 0) out vec4 Accum;
layout(location = 1) out float Reveal;


// PBR material parameters
uniform vec3 u_AmbientColor;
uniform float u_Metallic;
uniform float u_Roughness;
uniform sampler2D u_MetallicRoughnessTexture;
uniform sampler2D u_AlbedoTexture;
uniform sampler2D u_AOMap;

uniform int u_EnableMetallicRoughnessTexture;
uniform int u_EnableAlbedoTexture;

const float PI = 3.14159265359;
  
float DistributionGGX(vec3 N, vec3 H, float roughness)
{
    float a      = roughness*roughness;
    float a2     = a*a;
    float NdotH  = max(dot(N, H), 0.0);
    float NdotH2 = NdotH*NdotH;
	
    float num   = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;
	
    return num / denom;
}

float GeometrySchlickGGX(float NdotV, float roughness)
{
    float r = (roughness + 1.0);
    float k = (r*r) / 8.0;

    float num   = NdotV;
    float denom = NdotV * (1.0 - k) + k;
	
    return num / denom;
}

float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness)
{
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2  = GeometrySchlickGGX(NdotV, roughness);
    float ggx1  = GeometrySchlickGGX(NdotL, roughness);
	
    return ggx1 * ggx2;
}

vec3 fresnelSchlick(float cosTheta, vec3 F0)
{
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}  

vec2 CalcScreenTexCoord()
{
    return gl_FragCoord.xy / u_screenSize;
}


void main()
{		

    float ao = 1.0;
    float metallic = u_Metallic;
    float roughness = u_Roughness;
    vec3 albedo = v_Color.rgb;
    float alpha = v_Color.a;

    float z = gl_FragCoord.z;
    float weight = alpha * max(0.1, 3000.0 * pow(1.0 - z, 2.0));
    // float weight = 1.0;

    if (v_simpleRender == 0){


        if (u_EnableAlbedoTexture == 1 && u_renderMode == 3 && v_simpleRender == 0) {
            vec4 cv4 = texture(u_AlbedoTexture, v_Texcoord);
            albedo = cv4.rgb;
            alpha = cv4.a;
        }
        
        // Ensure alpha is sensible
        alpha = clamp(alpha, 0.0, 1.0);

        if (u_EnableMetallicRoughnessTexture == 1) {
            vec4 metallicRoughness = texture(u_MetallicRoughnessTexture, v_Texcoord);
            metallic = metallicRoughness.r;
            roughness = metallicRoughness.g;
        }

        if (u_enableAO == 1) {
            vec2 screenCoord = CalcScreenTexCoord();
            ao = texture(u_AOMap, screenCoord).r;
        }

        metallic = clamp(metallic, 0.001, 0.999);
        roughness = clamp(roughness, 0.001, 0.999);


        
        vec3 N;
        if (u_FlatShading == 1) {
            N = normalize(cross(dFdx(v_Position), dFdy(v_Position)));
        } else {
            N = normalize(v_Normal);
            if (!gl_FrontFacing) {
                N = -N;
            }
        }

        vec3 V = normalize(v_WorldSpaceCamPos - v_Position);


        if (u_renderMode == 3 || u_renderMode == 1){
            


            vec3 F0 = vec3(0.04); 
            F0 = mix(F0, albedo, metallic);
                    
            // reflectance equation
            vec3 Lo = vec3(0.0);
            for(int i = 0; i < u_NumLights; ++i) 
            {
                // calculate per-light radiance
                vec3 L = normalize(u_Lights[i].position - v_Position);
                vec3 H = normalize(V + L);
                // float distance    = length(u_Lights[i].position - v_Position);
                float distance = 0.6;
                float attenuation = 1.0 / (distance * distance);
                vec3 radiance     = u_Lights[i].color * attenuation;

                // cook-torrance brdf
                float NDF = DistributionGGX(N, H, roughness);        
                float G   = GeometrySmith(N, V, L, roughness);      
                vec3 F    = fresnelSchlick(max(dot(H, V), 0.0), F0);       
                
                vec3 kS = F;
                vec3 kD = vec3(1.0) - kS;
                kD *= 1.0 - metallic;	  
                
                vec3 numerator    = NDF * G * F;
                float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001;
                vec3 specular     = numerator / denominator;  
                    
                // add to outgoing radiance Lo
                float NdotL = max(dot(N, L), 0.0);                
                Lo += (kD * albedo / PI + specular) * radiance * NdotL; 
            }   
        
            vec3 ambient = u_AmbientColor * albedo * ao;  
            vec3 color = ambient + Lo;
            
            // gamma correction
            // color = color / (color + vec3(1.0));
            // color = pow(color, vec3(1.0/2.2));  
            
            // Weighted OIT output
            Accum = vec4(color * alpha, alpha) * weight;
            Reveal = alpha;


        }
        // render mode normal
        else if (u_renderMode == 2){
            // FragColor = vec4((1.0-N)*0.4 + 0.2, 1.0);
            Accum = vec4(N, 1.0);
            Reveal = alpha;
        }
        // render mode ao
        else if (u_renderMode == 4 && u_enableAO == 1){
            vec3 color = vec3(ao, ao, ao);
            Accum = vec4(color * alpha, alpha) * weight;
            Reveal = alpha;
        }

        else if (u_renderMode == 0){
            vec3 rgb = min(v_Color.rgb, vec3(1.0));
            Accum = vec4(rgb * alpha, alpha) * weight;
            Reveal = alpha;
        }

        else{
            vec3 rgb = min(v_Color.rgb, vec3(1.0));
            Accum = vec4(rgb * alpha, alpha) * weight;
            Reveal = alpha;
        }

    }

    else{
        // Simple Render
        // FragColor = v_Color;
        Accum = vec4(v_Color.rgb * alpha, alpha) * weight;
        Reveal = alpha;
    }
}  
