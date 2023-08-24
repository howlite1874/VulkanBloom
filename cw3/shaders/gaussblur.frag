#version 450

precision highp float;

layout (set = 0, binding = 0) uniform UBO {
    float blurScale;
    float blurStrength;
} ubo;

layout (set = 0, binding = 1) uniform sampler2D samplerColor;

layout (constant_id = 0) const int blurdirection = 1;

layout (location = 0) in vec2 inUV;

layout (location = 0) out vec4 outFragColor;

const int KERNEL_SIZE = 9;
const int LOOPS = 3;
const float weights[KERNEL_SIZE] = float[](
    0.0093, 0.028002, 0.065984, 0.121703, 0.175713, 0.121703, 0.065984, 0.028002, 0.0093
);

void main() 
{
    vec3 color = texture(samplerColor, inUV).rgb;

    vec2 tex_offset = 1.0 / textureSize(samplerColor, 0) * ubo.blurScale; // gets size of single texel
    vec3 result = texture(samplerColor, inUV).rgb * weights[4]; // current fragment's contribution

    for (int j = 1; j <= 4; ++j)
    {
        for (int i = 1; i <= LOOPS; ++i)
        {
            vec2 offset = tex_offset * float(i * j);
            if (blurdirection == 1)
            {
                // H
                result += texture(samplerColor, inUV + vec2(offset.x, 0.0)).rgb * weights[4 - j] * ubo.blurStrength;
                result += texture(samplerColor, inUV - vec2(offset.x, 0.0)).rgb * weights[4 - j] * ubo.blurStrength;
            }
            else
            {
                // V
                result += texture(samplerColor, inUV + vec2(0.0, offset.y)).rgb * weights[4 - j] * ubo.blurStrength;
                result += texture(samplerColor, inUV - vec2(0.0, offset.y)).rgb * weights[4 - j] * ubo.blurStrength;
            }
        }
    }

    outFragColor = vec4(result, 1.0);
}
