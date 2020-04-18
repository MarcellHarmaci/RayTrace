#include "framework.h"

struct Material {
	vec3 ka, kd, ks;
	float  shininess;
	Material(vec3 _kd, vec3 _ks, float _shininess): ka(_kd * M_PI), kd(_kd), ks(_ks) {
		shininess = _shininess;
	}
};

struct Hit {
	float t;
	vec3 position, normal;
	Material* material;
	Hit() { t = -1; }
};

struct Ray {
	vec3 start, dir;
	Ray(vec3 _start, vec3 _dir) {
		start = _start;
		dir = normalize(_dir);
	}
};

class Intersectable {
protected:
	Material* material;
public:
	virtual Hit intersect(const Ray& ray) = 0;
};

struct Quadrics : public Intersectable {
	mat4 Q;
	//vec3 param;

	Quadrics(){}
	Quadrics(mat4 _Q) {//, vec3 _param) { 
		Q = _Q;
		//param = _param; 
	}

	// Source: Homework assignment video
	float f(vec4 r) {
		return dot(r * Q, r);
	}

	// Source: Homework assignment video
	vec3 gradf(vec4 r) {
		vec4 g = r * Q * 2;
		return vec3(g.x, g.y, g.z);
	}
	
	vec2 solve(const Ray& ray) {
		vec4 p = vec4(ray.start.x, ray.start.y, ray.start.z, 1.0f);
		vec4 u = vec4(ray.dir.x, ray.dir.y, ray.dir.z, 0.0f);
		//printf("p1: %3.2f, p2: %3.2f, p3: %3.2f, p4: %3.2f\n", p.x, p.y, p.z, p.w);
		//printf("u1: %3.2f, u2: %3.2f, u3: %3.2f, u4: %3.2f\n", u.x, u.y, u.z, u.w);
		//printf("Q:\n%3.2f\t%3.2f\t%3.2f\t%3.2f\n%3.2f\t%3.2f\t%3.2f\t%3.2f\n%3.2f\t%3.2f\t%3.2f\t%3.2f\n%3.2f\t%3.2f\t%3.2f\t%3.2f\n",
		//	Q[0].x, Q[0].y, Q[0].z, Q[0].w,
		//	Q[1].x, Q[1].y, Q[1].z, Q[1].w,
		//	Q[2].x, Q[2].y, Q[2].z, Q[2].w,
		//	Q[3].x, Q[3].y, Q[3].z, Q[3].w);

		float a = dot(u * Q, u);
		float b = dot(u * Q, p);
		float c = dot(p * Q, p);

		float discr = b * b - a * c;
		if (discr < 0)
			return vec2(-1.0f, -1.0f);
	
		float sqrt_discr = sqrtf(discr);
		float t1 = (-b + sqrt_discr) / a;	// t1 >= t2 for sure
		float t2 = (-b - sqrt_discr) / a;

		return vec2(t1, t2);
	}

	Hit intersect(const Ray& ray) {
		Hit hit;

		vec2 solution = solve(ray);
		float t1 = solution.x;
		float t2 = solution.y;
		if (t1 <= 0) return hit;

		hit.t = (t2 > 0) ? t2 : t1;
		hit.position = ray.start + ray.dir * hit.t;
		hit.normal = normalize(gradf(vec4(hit.position.x, hit.position.y, hit.position.z, 1.0f)));
		hit.material = material;
		return hit;
	}
};

struct Hiperboloid : public Intersectable {
	vec3 param;
	mat4 mx;

	Hiperboloid(vec3 _param) {
		param = _param;
		mx = mat4(
			1.0f / (param.x * param.x), 0.0f, 0.0f, 0.0f,
			0.0f, 1.0f / (param.y * param.y), 0.0f, 0.0f,
			0.0f, 0.0f, -1.0f / (param.z * param.z), 0.0f,
			0.0f, 0.0f, 0.0f, -1.0f
		);
	}
};

struct Hyperboloid : public Intersectable {
	vec3 param;
	Quadrics mx;

	Hyperboloid(float _a, float _b, float _c, Material* _material) {
		param = vec3(_a, _b, _c);
		material = _material;
		mx = Quadrics(mat4(
			1.0f / (param.x * param.x), 0.0f, 0.0f, 0.0f,
			0.0f, 1.0f / (param.y * param.y), 0.0f, 0.0f,
			0.0f, 0.0f, -1.0f / (param.z * param.z), 0.0f,
			0.0f, 0.0f, 0.0f, -1.0f
		));
	}

	Hit intersect(const Ray& ray) {
		Hit hit;
		
		vec2 solution = mx.solve(ray);
		float t1 = solution.x;
		float t2 = solution.y;
		if (t1 <= 0) return hit;

		hit.t = (t2 > 0) ? t2 : t1;
		hit.position = ray.start + ray.dir * hit.t;
		hit.normal = normalize(mx.gradf(vec4(hit.position.x, hit.position.y, hit.position.z, 1.0f)));
		hit.material = material;
		return hit;
	}
};

struct Ellipsoid : public Intersectable {
	float a, b, c; // distances from center
	Quadrics quad;

	Ellipsoid(float _a, float _b, float _c, Material* _material) {
		a = _a;
		b = _b;
		c = _c;
		material = _material;
		quad = Quadrics(mat4(
			1.0f / (a * a), 0.0f, 0.0f, 0.0f,
			0.0f, 1.0f / (b * b), 0.0f, 0.0f,
			0.0f, 0.0f, 1.0f / (c * c), 0.0f,
			0.0f, 0.0f, 0.0f, -1.0f
		));
	}

	Hit intersect(const Ray& ray) {
		Hit hit;
		/*
		float d1 = ray.dir.x;
		float d2 = ray.dir.y;
		float d3 = ray.dir.z;

		float s1 = ray.start.x;
		float s2 = ray.start.y;
		float s3 = ray.start.z;

		float p1 = center.x;
		float p2 = center.y;
		float p3 = center.z;

		float b2c2 = b * b * c * c;
		float a2c2 = a * a * c * c;
		float a2b2 = a * a * b * b;

		float A = (d1 * d1 * b2c2) + (d2 * d2 * a2c2) + (d3 * d3 * a2b2);
		float B = 2 * (
			(d1 * (s1 - p1) * b2c2) +
			(d2 * (s2 - p2) * a2c2) +
			(d3 * (s3 - p3) * a2b2)
			);
		float C = -(a * a * b * b * c * c) +
			(s1 - p1) * (s1 - p1) * b2c2 +
			(s2 - p2) * (s2 - p2) * a2c2 +
			(s3 - p3) * (s3 - p3) * a2b2;

		float discr = B * B - 4.0f * A * C;
		if (discr < 0) return hit;
		float sqrt_discr = sqrtf(discr);
		float t1 = (-B + sqrt_discr) / 2.0f / A;	// t1 >= t2 for sure
		float t2 = (-B - sqrt_discr) / 2.0f / A;
		*/
		//hit.normal.x = 2.0f * (hit.position.x - center.x) / (a * a);
		//hit.normal.y = 2.0f * (hit.position.y - center.y) / (b * b);
		//hit.normal.z = 2.0f * (hit.position.z - center.z) / (c * c);
		//hit.normal = normalize(hit.normal);

		vec2 solution = quad.solve(ray);
		float t1 = solution.x;
		float t2 = solution.y;
		
		if (t1 <= 0) return hit;
		hit.t = (t2 > 0) ? t2 : t1;
		hit.position = ray.start + ray.dir * hit.t;
		hit.normal = normalize(quad.gradf(vec4(hit.position.x, hit.position.y, hit.position.z, 1.0f)));
		//hit.normal = -1.0f * hit.normal;

		hit.material = material;
		return hit;
	}
};

struct Sphere : public Intersectable {
	vec3 center;
	float radius;

	Sphere(const vec3& _center, float _radius, Material* _material) {
		center = _center;
		radius = _radius;
		material = _material;
	}

	Hit intersect(const Ray& ray) {
		Hit hit;
		vec3 dist = ray.start - center;
		float a = dot(ray.dir, ray.dir);
		float b = dot(dist, ray.dir) * 2.0f;
		float c = dot(dist, dist) - radius * radius;
		float discr = b * b - 4.0f * a * c;
		if (discr < 0) return hit;
		float sqrt_discr = sqrtf(discr);
		float t1 = (-b + sqrt_discr) / 2.0f / a;	// t1 >= t2 for sure
		float t2 = (-b - sqrt_discr) / 2.0f / a;
		if (t1 <= 0) return hit;
		hit.t = (t2 > 0) ? t2 : t1;
		hit.position = ray.start + ray.dir * hit.t;
		hit.normal = (hit.position - center) * (1.0f / radius);
		hit.material = material;
		return hit;
	}
};

class Camera {
	vec3 eye, lookat, right, up;
public:
	void set(vec3 _eye, vec3 _lookat, vec3 vup, float fov) {
		eye = _eye;
		lookat = _lookat;
		vec3 w = eye - lookat;
		float focus = length(w);
		right = normalize(cross(vup, w)) * focus * tanf(fov / 2);
		up = normalize(cross(w, right)) * focus * tanf(fov / 2);
	}
	Ray getRay(int X, int Y) {
		vec3 dir = lookat + right * (2.0f * (X + 0.5f) / windowWidth - 1) + up * (2.0f * (Y + 0.5f) / windowHeight - 1) - eye;
		return Ray(eye, dir);
	}
};

struct Light {
	vec3 direction;
	vec3 Le;
	Light(vec3 _direction, vec3 _Le) {
		direction = normalize(_direction);
		Le = _Le;
	}
};

float rnd() { return (float)rand() / RAND_MAX; }

const float epsilon = 0.0001f;

class Scene {
	std::vector<Intersectable *> objects;
	std::vector<Light *> lights;
	Camera camera;
	vec3 La;
public:
	void build() {
		vec3 eye = vec3(0.0f, 0.0f, 2.0f), vup = vec3(0, 1, 0), lookat = vec3(0, 0, 0);
		float fov = 45 * M_PI / 180;
		camera.set(eye, lookat, vup, fov);

		La = vec3(0.4f, 0.4f, 0.4f);
		// La = vec3(135.0f/255.0f, 206.0f / 255.0f, 235.0f / 255.0f); Sky blue
		vec3 lightDirection(0.0f, 0.5f, 0.8f), Le(2, 2, 2);
		lights.push_back(new Light(lightDirection, Le));

		vec3 kd1(0.3f, 0.2f, 0.1f), kd2(0.1f, 0.2f, 0.3f), ks(2, 2, 2);
		Material * material1 = new Material(kd1, ks, 50);
		Material * material2 = new Material(kd2, ks, 50);

		for (int i = 0; i < 50; i++) {
			objects.push_back(new Sphere(vec3(rnd() - 0.5f, rnd() - 0.5f, rnd() - 0.5f), rnd() * 0.1f, material2));
		}
		objects.push_back(new Ellipsoid(0.3f, 0.5f, 0.4f, material1));
		objects.push_back(new Hyperboloid(0.45f, 0.65f, 0.55f, material1));
	}

	void render(std::vector<vec4>& image) {
		for (int Y = 0; Y < windowHeight; Y++) {
#pragma omp parallel for
			for (int X = 0; X < windowWidth; X++) {
				vec3 color = trace(camera.getRay(X, Y));
				image[Y * windowWidth + X] = vec4(color.x, color.y, color.z, 1);
			}
		}
	}

	Hit firstIntersect(Ray ray) {
		Hit bestHit;
		for (Intersectable * object : objects) {
			Hit hit = object->intersect(ray); //  hit.t < 0 if no intersection
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
		}
		if (dot(ray.dir, bestHit.normal) > 0)
			bestHit.normal = bestHit.normal * (-1);
		return bestHit;
	}

	bool shadowIntersect(Ray ray) {	// for directional lights
		for (Intersectable * object : objects) 
			if (object->intersect(ray).t > 0) return true;

		return false;
	}

	vec3 trace(Ray ray, int depth = 0) {
		Hit hit = firstIntersect(ray);
		if (hit.t < 0) 
			return La;

		vec3 outRadiance = hit.material->ka * La;

		for (Light * light : lights) {
			Ray shadowRay(hit.position + hit.normal * epsilon, light->direction);
			float cosTheta = dot(hit.normal, light->direction);

			if (cosTheta > 0 && !shadowIntersect(shadowRay)) {	// shadow computation
				outRadiance = outRadiance + light->Le * hit.material->kd * cosTheta;
				vec3 halfway = normalize(-ray.dir + light->direction);
				float cosDelta = dot(hit.normal, halfway);
				if (cosDelta > 0) 
					outRadiance = outRadiance + light->Le * hit.material->ks * powf(cosDelta, hit.material->shininess);
			}
		}
		return outRadiance;
	}
};

GPUProgram gpuProgram; // vertex and fragment shaders
Scene scene;

// vertex shader in GLSL
const char *vertexSource = R"(
	#version 330
    precision highp float;

	layout(location = 0) in vec2 cVertexPosition;	// Attrib Array 0
	out vec2 texcoord;

	void main() {
		texcoord = (cVertexPosition + vec2(1, 1))/2;							// -1,1 to 0,1
		gl_Position = vec4(cVertexPosition.x, cVertexPosition.y, 0, 1); 		// transform to clipping space
	}
)";

// fragment shader in GLSL
const char *fragmentSource = R"(
	#version 330
    precision highp float;

	uniform sampler2D textureUnit;
	in  vec2 texcoord;			// interpolated texture coordinates
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		fragmentColor = texture(textureUnit, texcoord); 
	}
)";

class FullScreenTexturedQuad {
	unsigned int vao;	// vertex array object id and texture id
	Texture texture;
public:
	FullScreenTexturedQuad(int windowWidth, int windowHeight, std::vector<vec4>& image)
		: texture(windowWidth, windowHeight, image) 
	{
		glGenVertexArrays(1, &vao);	// create 1 vertex array object
		glBindVertexArray(vao);		// make it active

		unsigned int vbo;		// vertex buffer objects
		glGenBuffers(1, &vbo);	// Generate 1 vertex buffer objects

		// vertex coordinates: vbo0 -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo); // make it active, it is an array
		float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };	// two triangles forming a quad
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified 
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);     // stride and offset: it is tightly packed
	}

	void Draw() {
		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		gpuProgram.setUniform(texture, "textureUnit");
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);	// draw two triangles forming a quad
	}
};

FullScreenTexturedQuad * fullScreenTexturedQuad;

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	scene.build();

	std::vector<vec4> image(windowWidth * windowHeight);
	long timeStart = glutGet(GLUT_ELAPSED_TIME);
	scene.render(image);
	long timeEnd = glutGet(GLUT_ELAPSED_TIME);
	printf("Rendering time: %d milliseconds\n", (timeEnd - timeStart));

	// copy image to GPU as a texture
	fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight, image);

	// create program for the GPU
	gpuProgram.create(vertexSource, fragmentSource, "fragmentColor");
}

// Window has become invalid: Redraw
void onDisplay() {
	fullScreenTexturedQuad->Draw();
	glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {

}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
}
