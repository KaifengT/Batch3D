// reference from https://john-chapman-graphics.blogspot.com/2013/01/ssao-tutorial.html
// reference from https://learnopengl.com/Advanced-Lighting/SSAO, https://blog.csdn.net/u013617851/article/details/122397080

#version 460 core

in vec2 TexCoord;

out vec4 FragColor;

uniform sampler2D u_positionMap;
uniform sampler2D u_normalMap;
uniform sampler2D u_kernelNoise;
uniform vec2 u_screenSize;

uniform mat4 u_ProjMatrix;

const int MAX_KERNEL_SIZE = 256;
uniform vec3 u_kernel[MAX_KERNEL_SIZE];
uniform int u_kernelSize;
uniform int u_projMode;

const float bias = 0.025;

uniform float u_radiusPixels; // radius in pixels, used to calculate view space radius

void main()
{
    vec2 noiseScale = u_screenSize / 4.0;

    vec3 Pos = texture(u_positionMap, TexCoord).xyz;
    vec3 Normal = texture(u_normalMap, TexCoord).xyz;
    vec3 Noise = texture(u_kernelNoise, TexCoord * noiseScale).xyz;

    vec3 tangent = normalize(Noise - Normal * dot(Noise, Normal));
    vec3 bitangent = normalize(cross(Normal, tangent));
    mat3 TBN = mat3(tangent, bitangent, Normal);


    
    float fy = u_ProjMatrix[1][1];
    float z = 1.0;
    if (u_projMode == 0){
        z = - Pos.z;
    }
    else {
        z = 1.0;
    }
    
    float viewRadius = (u_radiusPixels * 2.0 * z) / (u_screenSize.y * fy);


    float AO = 0.0;

    for (int i = 0 ; i < u_kernelSize ; i++) {

        vec3 samplePos = Pos + TBN * (u_kernel[i] * viewRadius);



        vec4 offset = vec4(samplePos, 1.0);
        offset = u_ProjMatrix * offset;
        
        offset.xyz /= offset.w;

        offset.xyz = offset.xyz * 0.5 + 0.5;
            
        float sampleDepth = texture(u_positionMap, offset.xy).z;

        float rangeCheck = smoothstep(0.0, 1.0, viewRadius / abs(Pos.z - sampleDepth) + 1e-4);
        AO += (sampleDepth >= samplePos.z + bias ? 1.0 : 0.0) * rangeCheck;

    }

    AO = 1.0 - (AO/float(u_kernelSize));
    // AO = 0.5;
    // vec3 A0_vec3 = vec3(pow(AO, 2.0));

    // FragColor = vec4(A0_vec3, 1.0);
    FragColor = vec4(AO);
    // FragColor = vec3(AO);
    // FragColor = AO;
    // FragColor = 1.0;
    // FragColor = vec4(AO, AO, AO, 1.0);
    // FragColor = vec4(Noise, 1.0);

}