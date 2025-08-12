#version 120
varying vec2 TexCoord;
#define FragColor gl_FragColor

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
    vec3 centerPos = texture2D(u_PositionMap, TexCoord).xyz;
    vec3 centerN   = normalize(texture2D(u_NormalMap,   TexCoord).xyz);
    float centerAO = texture2D(u_AOMap, TexCoord).r;

    if (!all(greaterThanEqual(abs(centerPos), vec3(0.0)))) {
        FragColor = vec4(centerAO, centerAO, centerAO, 1.0);
        return;
    }

    float sumW = 0.0;
    float aoAccum = 0.0;

    for (int y = -u_Radius; y <= u_Radius; ++y) {
        for (int x = -u_Radius; x <= u_Radius; ++x) {

            vec2 offset = vec2(float(x), float(y)) * u_TexelSize;
            vec2 tc = TexCoord + offset;

            vec3 samplePos = texture2D(u_PositionMap, tc).xyz;
            vec3 sampleN   = normalize(texture2D(u_NormalMap, tc).xyz);
            float sampleAO = texture2D(u_AOMap, tc).r;

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