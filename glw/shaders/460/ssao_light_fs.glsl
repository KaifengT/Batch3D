
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

uniform vec2 u_screenSize;
uniform int u_enableAO;

out vec4 FragColor;


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




    if (v_simpleRender == 0){


        if (u_EnableAlbedoTexture == 1 && u_renderMode == 3 && v_simpleRender == 0) {
            vec4 cv4 = texture(u_AlbedoTexture, v_Texcoord);
            albedo = cv4.rgb;
            alpha = cv4.a;
        }

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

        vec3 N = normalize(v_Normal);
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
        
            FragColor = vec4(color, alpha);
            // FragColor = vec4(metallic, roughness, 0.0, 1.0);


        }
        // render mode normal
        else if (u_renderMode == 2){
            // FragColor = vec4((1.0-N)*0.4 + 0.2, 1.0);
            FragColor = vec4(N, 1.0);
        }
        // render mode ao
        else if (u_renderMode == 4 && u_enableAO == 1){
            FragColor = vec4(ao, ao, ao, 1.0);
        }

        else if (u_renderMode == 0){
            
            vec3 rgb = min(v_Color.rgb, vec3(1.0));
            FragColor = vec4(rgb, alpha);
        }

        else{
            vec3 rgb = min(v_Color.rgb, vec3(1.0));
            FragColor = vec4(rgb, alpha);
        }



    }

    else{

        // vec2 coord = gl_PointCoord - vec2(0.5);
        // if (length(coord) > 0.5) {
        //     discard;
        // }

        FragColor = v_Color;

    }
}  