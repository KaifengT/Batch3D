// #version 330

// in vec2 TexCoord;

// out vec4 FragColor;

// uniform sampler2D u_AOMap;

// float Offsets[4] = float[]( -1.5, -0.5, 0.5, 1.5 );

// void main()
// {
//     vec3 Color = vec3(0.0, 0.0, 0.0);

//     for (int i = 0 ; i < 4 ; i++) {
//         for (int j = 0 ; j < 4 ; j++) {
//             vec2 tc = TexCoord;
//             tc.x = TexCoord.x + Offsets[j] / textureSize(u_AOMap, 0).x;
//             tc.y = TexCoord.y + Offsets[i] / textureSize(u_AOMap, 0).y;
//             Color += texture(u_AOMap, tc).xyz;
//         }
//     }

//     Color /= 16.0;

//     FragColor = vec4(Color, 1.0);
// }



#version 330
in vec2 TexCoord;
out vec4 FragColor;

uniform sampler2D u_AOMap;
uniform sampler2D u_PositionMap;
uniform sampler2D u_NormalMap;
uniform vec2 u_TexelSize;

uniform float u_SpatialSigma;
uniform float u_DepthSigma;
uniform float u_NormalSigma;
uniform int   u_Radius;

float spatialWeight(int dx, int dy) {
    float r2 = float(dx*dx + dy*dy);
    float denom = 2.0 * u_SpatialSigma * u_SpatialSigma;
    return exp(-r2 / denom);
}

void main() {
    vec3 centerPos = texture(u_PositionMap, TexCoord).xyz;
    vec3 centerN   = normalize(texture(u_NormalMap,   TexCoord).xyz);
    float centerAO = texture(u_AOMap, TexCoord).r;

    // Optional: w==0
    if (!all(greaterThanEqual(abs(centerPos), vec3(0.0)))) {
        FragColor = vec4(centerAO, centerAO, centerAO, 1.0);
        return;
    }

    float sumW = 0.0;
    float aoAccum = 0.0;

    for (int y = -u_Radius; y <= u_Radius; ++y) {
        for (int x = -u_Radius; x <= u_Radius; ++x) {

            vec2 offset = vec2(x, y) * u_TexelSize;
            vec2 tc = TexCoord + offset;

            vec3 samplePos = texture(u_PositionMap, tc).xyz;
            vec3 sampleN   = normalize(texture(u_NormalMap, tc).xyz);
            float sampleAO = texture(u_AOMap, tc).r;


            if (!all(greaterThanEqual(abs(samplePos), vec3(0.0)))) {
                continue;
            }

            float wSpatial = spatialWeight(x, y);

            float dz = samplePos.z - centerPos.z;
            float wDepth = exp(-(dz*dz) / (2.0 * u_DepthSigma * u_DepthSigma));

            float ndot = max(dot(centerN, sampleN), 0.0);
            float normalDiff = 1.0 - ndot;
            float wNormal = exp(-(normalDiff * normalDiff) * u_NormalSigma);

            float w = wSpatial * wDepth * wNormal;

            aoAccum += sampleAO * w;
            sumW += w;
        }
    }

    float ao = (sumW > 0.0) ? (aoAccum / sumW) : centerAO;
    ao = clamp(ao, 0.0, 1.0);

    FragColor = vec4(vec3(ao), 1.0);
}

