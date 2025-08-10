#version 330

in vec2 TexCoord;

out vec4 FragColor;

uniform sampler2D gPositionMap;
uniform sampler2D gNormalMap;

uniform float gSampleRad;
uniform mat4 gProj;

const int MAX_KERNEL_SIZE = 128;
uniform vec3 gKernel[MAX_KERNEL_SIZE];


void main()
{
    vec3 Pos = texture(gPositionMap, TexCoord).xyz;
    vec3 Normal = texture(gNormalMap, TexCoord).xyz;

    float AO = 0.0;

    for (int i = 0 ; i < MAX_KERNEL_SIZE ; i++) {
        vec3 samplePos = Pos + gKernel[i];
        vec4 offset = vec4(samplePos, 1.0);
        offset = gProj * offset;
        offset.xy /= offset.w;
        offset.xy = offset.xy * 0.5 + vec2(0.5);
            
        float sampleDepth = texture(gPositionMap, offset.xy).b;

        if (abs(Pos.z - sampleDepth) < gSampleRad) {
            AO += step(sampleDepth,samplePos.z);
        }
    }

    AO = 1.0 - AO/128.0;
    // AO = 0.5;
    // vec3 A0_vec3 = vec3(pow(AO, 2.0));

    // FragColor = vec4(A0_vec3, 1.0);
    FragColor = vec4(pow(AO, 2.0));

}