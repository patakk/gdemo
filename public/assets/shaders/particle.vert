attribute vec4 color;
attribute vec2 size;
attribute float angle;
attribute float index;

varying vec4 vColor;
varying vec2 vSize;
varying float vAngle;
varying float vIndex;

uniform float u_time;
uniform float u_spread;
uniform float u_seed;
uniform vec2 u_mouse;
uniform vec4 u_diffuse;
uniform float u_scrollscale;
uniform float u_winscale;

float randomNoise(vec2 p) {
  return fract(16791.414*sin(7.*p.x+p.y*73.41));
}

float random (in vec2 _st) {
    return fract(sin(dot(_st.xy,
                         vec2(12.9898,78.233)))*
        43758.5453123);
}

float noise (in vec2 _st) {
    vec2 i = floor(_st);
    vec2 f = fract(_st);

    // Four corners in 2D of a tile
    float a = random(i);
    float b = random(i + vec2(1.0, 0.0));
    float c = random(i + vec2(0.0, 1.0));
    float d = random(i + vec2(1.0, 1.0));

    vec2 u = f * f * (3.0 - 2.0 * f);

    return mix(a, b, u.x) +
            (c - a)* u.y * (1.0 - u.x) +
            (d - b) * u.x * u.y;
}

float noise3 (in vec2 _st, in float t) {
    vec2 i = floor(_st+t);
    vec2 f = fract(_st+t);

    // Four corners in 2D of a tile
    float a = random(i);
    float b = random(i + vec2(1.0, 0.0));
    float c = random(i + vec2(0.0, 1.0));
    float d = random(i + vec2(1.0, 1.0));

    vec2 u = f * f * (3.0 - 2.0 * f);

    return mix(a, b, u.x) +
            (c - a)* u.y * (1.0 - u.x) +
            (d - b) * u.x * u.y;
}

#define NUM_OCTAVES 5

float fbm ( in vec2 _st) {
    float v = 0.0;
    float a = 0.5;
    vec2 shift = vec2(100.0);
    // Rotate to reduce axial bias
    mat2 rot = mat2(cos(0.5), sin(0.5),
                    -sin(0.5), cos(0.50));
    for (int i = 0; i < NUM_OCTAVES; ++i) {
        v += a * noise(_st);
        _st = rot * _st * 2.0 + shift;
        a *= 0.5;
    }
    return v;
}

float fbm3 ( in vec2 _st, in float t) {
    float v = 0.0;
    float a = 0.5;
    vec2 shift = vec2(100.0);
    // Rotate to reduce axial bias
    mat2 rot = mat2(cos(0.5), sin(0.5),
                    -sin(0.5), cos(0.50));
    for (int i = 0; i < NUM_OCTAVES; ++i) {
        v += a * noise3(_st, t);
        _st = rot * _st * 2.0 + shift;
        a *= 0.5;
    }
    return v;
}


// gaussian blur filter modified from Filip S. at intel 
// https://software.intel.com/en-us/blogs/2014/07/15/an-investigation-of-fast-real-time-gpu-based-image-blur-algorithms
// this function takes three parameters, the texture we want to blur, the uvs, and the texelSize
vec3 gaussianBlur( sampler2D t, vec2 texUV, vec2 stepSize ){   
	// a variable for our output                                                                                                                                                                 
	vec3 colOut = vec3( 0.0 );                                                                                                                                   

	// stepCount is 9 because we have 9 items in our array , const means that 9 will never change and is required loops in glsl                                                                                                                                     
	const int stepCount = 9;

	// these weights were pulled from the link above
	float gWeights[stepCount];
	    gWeights[0] = 0.10855;
	    gWeights[1] = 0.13135;
	    gWeights[2] = 0.10406;
	    gWeights[3] = 0.07216;
	    gWeights[4] = 0.04380;
	    gWeights[5] = 0.02328;
	    gWeights[6] = 0.01083;
	    gWeights[7] = 0.00441;
	    gWeights[8] = 0.00157;

	// these offsets were also pulled from the link above
	float gOffsets[stepCount];
	    gOffsets[0] = 0.66293;
	    gOffsets[1] = 2.47904;
	    gOffsets[2] = 4.46232;
	    gOffsets[3] = 6.44568;
	    gOffsets[4] = 8.42917;
	    gOffsets[5] = 10.41281;
	    gOffsets[6] = 12.39664;
	    gOffsets[7] = 14.38070;
	    gOffsets[8] = 16.36501;
	
	// lets loop nine times
	for( int i = 0; i < stepCount; i++ ){  

		// multiply the texel size by the by the offset value                                                                                                                                                               
	    vec2 texCoordOffset = gOffsets[i] * stepSize;

		// sample to the left and to the right of the texture and add them together                                                                                                           
	    vec3 col = texture2D( t, texUV + texCoordOffset ).xyz + texture2D( t, texUV - texCoordOffset ).xyz; 

		// multiply col by the gaussian weight value from the array
		col *= gWeights[i];

		// add it all up
	    colOut +=  col;                                                                                                                               
	}

	// our final value is returned as col out
	return colOut;                                                                                                                                                   
} 

float fff(vec2 st, float seed){

    vec2 q = vec2(0.);
    q.x = fbm3( st + 0.1, seed*.11);
    q.y = fbm3( st + vec2(1.0), seed*.11);
    vec2 r = vec2(0.);
    r.x = fbm3( st + 1.0*q + vec2(1.7,9.2)+ 0.15*seed*0.11, seed*.11);
    r.y = fbm3( st + 1.0*q + vec2(8.3,2.8)+ 0.126*seed*0.11, seed*.11);
    float f = fbm3(st+r, seed*.11);
    float ff = (f+0.120*f*f+.5*f);

    return ff-.75;
}

vec3 rgb2hsv(vec3 c)
{
    vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));

    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

vec3 hsv2rgb(vec3 c)
{
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

float power(float p, float g) {
    if (p < 0.5)
        return 0.5 * pow(2.*p, g);
    else
        return 1. - 0.5 * pow(2.*(1. - p), g);
}

void main() {
    //vAlpha = alpha;

    vec3 individual = vec3(0., 0., 0.);

    individual.x = 10.*u_spread*(fff(vec2(31.31, 22.1)+index, .0*u_time+index*31.31+55.2214));
    individual.y = 10.*u_spread*(fff(vec2(83.31, 65.1)+index, .0*u_time+index*31.31+22.651));


    vec4 mvPosition = modelViewMatrix * vec4( position+individual+vec3(u_mouse*(vec2(1.,-1.)), 0.0), 1.0 );


    float pscale = max(size.x, size.y)*4. + 0.*5.*sin(u_time/60.*5. + .074414313*index);
    gl_PointSize = .8* pscale * u_scrollscale * 1.5 * .5 * ((1.-.3) + 0.*2.*.3*random(mvPosition.xy+mod(u_time/100., 1.0)));
    //gl_PointSize = pscale * u_scrollscale * u_winscale * .5;
    gl_Position = projectionMatrix * mvPosition;

    // drawing animation
    //if(index/2250. > u_time)
    //    gl_PointSize = 0.;

    vec3 hsv = vec3(0.);
    vec3 hsvdiffuse = rgb2hsv(u_diffuse.rgb);

    hsv.r = mod(1.0 + hsvdiffuse.r + .4*0.*fff(vec2(u_time,u_time)*.0131, u_seed*.1+index), 1.0);
    hsv.g = min(hsvdiffuse.g*.99 + .0*fff(vec2(u_time,u_time)*.0431, u_seed*.21+index), 1.0);
    hsv.b = min(hsvdiffuse.b * (.8 + .3*smoothstep(.6, .1, fff(vec2(u_time,u_time)*.0431, u_seed*.21+index))), 1.0);
    
    
    if(mod(index, 3.5) < 0.01 && (pscale < 18.)){
        hsv.r = mod(hsv.r+0.1, 1.0);
        hsv.g = hsv.g*.15;
        hsv.b = min(1.0, 2.1*hsv.b);
    }
    
    if(mod(index+1., 3.5) < 0.01 && (pscale < 18.)){
        hsv.r = mod(hsv.r+0.1, 1.0);
        hsv.g = hsv.g*.5;
        hsv.b = .16*hsv.b;
    }
    

    vec3 rgb = hsv2rgb(hsv);


    float alpha = .03 - .0*power((fff(vec2(u_time,u_time)*1.11+0.*index, 31.31+u_seed*.1) + .5)*.68,1.);
    //vColor = vec4(rgb, power(alpha*6., 1.)/6. * u_diffuse.a);
    vColor = vec4(vec3(smoothstep(.1, .3, fff(vec2(u_time,u_time)*.0431, u_seed*.21+index))), .4);
    vColor = vec4(rgb, alpha);

    //vColor.r *= .5 + (1.-.5)*2.*random(mvPosition.xy+mod(u_time/100.+.366, 1.0));
    //vColor.g *= .5 + (1.-.5)*2.*random(mvPosition.xy+mod(u_time/100.+.253, 1.0));
    //vColor.b *= .5 + (1.-.5)*2.*random(mvPosition.xy+mod(u_time/100.+.112, 1.0));

    vSize = size;
    vAngle = angle;
    vIndex = index;
}