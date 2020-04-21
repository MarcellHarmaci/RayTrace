#include "framework.h"

enum MaterialType {
	ROUGH, REFLECTIVE
};

struct Material {
	vec3 ka, kd, ks;
	float  shininess;
	vec3 F0;
	MaterialType type;
	Material(MaterialType _type) {
		type = _type;
	}
};

struct RoughMaterial : Material {
	RoughMaterial(vec3 _kd, vec3 _ks, float _shininess) : Material(ROUGH) {
		ka = _kd * M_PI;
		kd = _kd;
		ks = _ks;
		shininess = _shininess;
	}
};

vec3 operator/(vec3 num, vec3 denom) {
	return vec3(num.x / denom.x, num.y / denom.y, num.z / denom.z);
}

struct ReflectiveMaterial : Material {
	ReflectiveMaterial(vec3 n, vec3 kappa) : Material(REFLECTIVE) {
		vec3 one(1, 1, 1);
		F0 = ((n - one) * (n - one) + kappa * kappa) /
			((n + one) * (n + one) + kappa * kappa);
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

mat4 transpose(mat4 mx) {
	vec4 a = mx[0], b = mx[1], c = mx[2], d = mx[3];
	return mat4(
		a.x, b.x, c.x, d.x,
		a.y, b.y, c.y, d.y,
		a.z, b.z, c.z, d.z,
		a.w, b.w, c.w, d.w
	);
}

mat4 invTranslateMx(mat4 mx) {
	vec4 param = mx[3];
	return mat4(
		1.0f, 0.0f, 0.0f, 0.0f,
		0.0f, 1.0f, 0.0f, 0.0f,
		0.0f, 0.0f, 1.0f, 0.0f,
		-param.x, -param.y, -param.z, 1.0f
	);
}

mat4 invScaleMx(mat4 mx) {
	vec3 param(mx[0].x, mx[1].y, mx[2].z);
	return mat4(
		1.0f / mx[0].x, 0.0f, 0.0f, 0.0f,
		0.0f, 1.0f / mx[1].y, 0.0f, 0.0f,
		0.0f, 0.0f, 1.0f / mx[2].z, 0.0f,
		0.0f, 0.0f, 0.0f, 1.0f
	);
}

mat4 invRotationMx(mat4 mx) {
	vec4 a = mx[0], b = mx[1], c = mx[2];
	float a3b2c1 = a.z * b.y * c.x, a2b3c1 = a.y * b.z * c.x, a3b1c2 = a.z * b.x * c.y, a1b3c2 = a.x * b.z * c.y, a2b1c3 = a.y * b.x * c.z, a1b2c3 = a.x * b.y * c.z;
	return mat4(
		vec4(
			(b.z * c.y - b.y * c.z) / (a3b2c1 - a2b3c1 - a3b1c2 + a1b3c2 + a2b1c3 - a1b2c3),
			(a.z * c.y - a.y * c.z) / (-a3b2c1 + a2b3c1 + a3b1c2 - a1b3c2 - a2b1c3 + a1b2c3),
			(a.z * b.y - a.y * b.z) / (a3b2c1 - a2b3c1 - a3b1c2 + a1b3c2 + a2b1c3 - a1b2c3),
			0.0f),
		vec4(
			(b.z * c.x - b.x * c.z) / (-a3b2c1 + a2b3c1 + a3b1c2 - a1b3c2 - a2b1c3 + a1b2c3),
			(a.z * c.x - a.x * c.z) / (a3b2c1 - a2b3c1 - a3b1c2 + a1b3c2 + a2b1c3 - a1b2c3),
			(a.z * b.x - a.x * b.z) / (-a3b2c1 + a2b3c1 + a3b1c2 - a1b3c2 - a2b1c3 + a1b2c3),
			0.0f),
		vec4(
			(b.y * c.x - b.x * c.y) / (a3b2c1 - a2b3c1 - a3b1c2 + a1b3c2 + a2b1c3 - a1b2c3),
			(a.y * c.x - a.x * c.y) / (-a3b2c1 + a2b3c1 + a3b1c2 - a1b3c2 - a2b1c3 + a1b2c3),
			(a.y * b.x - a.x * b.y) / (a3b2c1 - a2b3c1 - a3b1c2 + a1b3c2 + a2b1c3 - a1b2c3),
			0.0f),
		vec4(0.0f, 0.0f, 0.0f, 1.0f)
	);
}

class Quadric : public Intersectable {
protected:
	mat4 Q;

public:
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
		if (solution.x <= 0) return hit;

		hit.t = (solution.y > 0) ? solution.y : solution.x;
		hit.position = ray.start + ray.dir * hit.t;

		hit.normal = normalize(gradf(vec4(hit.position.x, hit.position.y, hit.position.z, 1.0f)));
		hit.material = material;
		return hit;
	}

	void translate(vec3 t) {
		mat4 trMx = TranslateMatrix(t);
		mat4 invTrMx = invTranslateMx(trMx);
		mat4 transposeOfInvTrMx = transpose(invTrMx);
		Q = invTrMx * Q * transposeOfInvTrMx;
	}

	void rotate(float angle, vec3 w) {
		mat4 rotaMx = RotationMatrix(angle, w);
		mat4 invRotaMx = invRotationMx(rotaMx);
		mat4 transposeOfInvRotaMx = transpose(invRotaMx);
		Q = invRotaMx * Q * transposeOfInvRotaMx;
	}
};

struct Hyperboloid : public Quadric {
	float a, b, c;
	vec3 top, bottom;

	Hyperboloid(float _a, float _b, float _c, Material* _material) {
		material = _material;
		a = _a; b = _b; c = _c;
		Q = mat4(
			1.0f / (a * a), 0.0f, 0.0f, 0.0f,
			0.0f, 1.0f / (b * b), 0.0f, 0.0f,
			0.0f, 0.0f, -1.0f / (c * c), 0.0f,
			0.0f, 0.0f, 0.0f, -1.0f
		);
		top = vec3(0, 1, 0); bottom = vec3(0, -1, 0);
	}

	void setEnds(vec3 _top, vec3 _bottom) {
		top = _top;
		bottom = _bottom;
	}

	Hit intersect(const Ray& ray) {
		Hit hit;
		vec3 nPos(0.0f, 1.0f, 0.0f), nNeg(0.0f, -1.0f, 0.0f);

		vec2 solution = solve(ray);
		if (solution.x <= 0) return hit;

		if (solution.y > 0) {
			hit.t = solution.y;
			hit.position = ray.start + ray.dir * hit.t;

			if (dot(nPos, hit.position - top) > 0 ||
				dot(nNeg, hit.position - bottom) > 0) {
				hit.t = solution.x;
				hit.position = ray.start + ray.dir * hit.t;

				if (dot(nPos, hit.position - top) > 0 ||
					dot(nNeg, hit.position - bottom) > 0)
					return Hit();
			}
		}
		else {
			hit.t = solution.x;
			hit.position = ray.start + ray.dir * hit.t;

			if (dot(nPos, hit.position - top) > 0 ||
				dot(nNeg, hit.position - bottom) > 0) {
				return Hit();
			}
		}

		hit.normal = normalize(gradf(vec4(hit.position.x, hit.position.y, hit.position.z, 1.0f)));
		hit.material = material;
		return hit;
	}
};

struct Cylinder : public Quadric {
	float a, b;

	Cylinder(float _a, float _b, Material* _material) {
		material = _material;
		a = _a; b = _b;
		Q = mat4(
			1.0f / (a * a), 0.0f, 0.0f, 0.0f,
			0.0f, 1.0f / (b * b), 0.0f, 0.0f,
			0.0f, 0.0f, 0.0f, 0.0f,
			0.0f, 0.0f, 0.0f, -1.0f
		);
	}

	Hit intersect(const Ray& ray) {
		Hit hit;
		vec3 nPos(0.0f, 0.0f, 1.0f), nNeg(0.0f, 0.0f, -1.0f);
		vec3 end1(0.0f, 0.0f, 0.5f), end2(0.0f, 0.0f, -0.4f);

		vec2 solution = solve(ray);
		if (solution.x <= 0) return hit;

		if (solution.y > 0) {
			hit.t = solution.y;
			hit.position = ray.start + ray.dir * hit.t;

			if (dot(nPos, hit.position - end1) > 0 ||
				dot(nNeg, hit.position - end2) > 0) {
				hit.t = solution.x;
				hit.position = ray.start + ray.dir * hit.t;

				if (dot(nPos, hit.position - end1) > 0 ||
					dot(nNeg, hit.position - end2) > 0)
					return Hit();
			}
		}
		else {
			hit.t = solution.x;
			hit.position = ray.start + ray.dir * hit.t;

			if (dot(nPos, hit.position - end1) > 0 ||
				dot(nNeg, hit.position - end2) > 0) {
				return Hit();
			}
		}

		hit.normal = normalize(gradf(vec4(hit.position.x, hit.position.y, hit.position.z, 1.0f)));
		hit.material = material;
		return hit;
	}
};

struct Ellipsoid : public Quadric {
	float a, b, c;
	bool isRoom;

	Ellipsoid(float _a, float _b, float _c, Material* _material) {
		a = _a; b = _b; c = _c;
		material = _material;
		Q = mat4(
			1.0f / (a * a), 0.0f, 0.0f, 0.0f,
			0.0f, 1.0f / (b * b), 0.0f, 0.0f,
			0.0f, 0.0f, 1.0f / (c * c), 0.0f,
			0.0f, 0.0f, 0.0f, -1.0f
		);
		isRoom = false;
	}

	void makeRoom() {
		isRoom = true;
	}

	Hit intersect(const Ray& ray) {
		Hit hit;

		vec2 solution = solve(ray);
		if (solution.x <= 0) return hit;

		hit.t = (solution.y > 0) ? solution.y : solution.x;
		hit.position = ray.start + ray.dir * hit.t;
		if (isRoom && dot(vec3(0.0f, 1.0f, 0.0f), hit.position - vec3(0.0f, 0.99f, 0.0f)) > 0) return Hit();

		hit.normal = normalize(gradf(vec4(hit.position.x, hit.position.y, hit.position.z, 1.0f)));
		hit.material = material;
		return hit;
	}
};

struct EllipticalCone : public Quadric {
	float a, b, c;

	EllipticalCone(float _a, float _b, float _c, Material* _material) {
		material = _material;
		a = _a; b = _b; c = _c;
		Q = mat4(
			1.0f / (a * a), 0.0f, 0.0f, 0.0f,
			0.0f, 1.0f / (b * b), 0.0f, 0.0f,
			0.0f, 0.0f, -1.0f / (c * c), 0.0f,
			0.0f, 0.0f, 0.0f, 0.0f
		);
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

struct Plane : public Intersectable {
	vec3 p0, normal;

	Plane(vec3 _p0, vec3 _normal, Material* _material) {
		p0 = _p0;
		normal = normalize(_normal);
		material = _material;
	}

	Hit intersect(const Ray& ray) {
		Hit hit;

		float t = dot(normal, (p0 - ray.start) / ray.dir);
		if (t < 0) return hit;

		hit.t = t;
		hit.position = ray.start + ray.dir * hit.t;
		hit.normal = normal;
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
	std::vector<Intersectable*> objects;
	std::vector<Light*> lights;
	std::vector<vec3> controlPoints;
	Camera camera;
	vec3 La;

public:
	void genControlPoints(float y, float r) {
		int cntGood = 0, cntBad = 0;
		for (int i = 0; i < 50; i++) {
			vec3 cp(2 * r * rnd() - r, y, 2 * r * rnd() - r);
			while (length(vec2(cp.x, cp.z)) > r) {
				cp = vec3(2 * r * rnd() - r, y, 2 * r * rnd() - r);
			}
			controlPoints.push_back(cp);
		}
	}

	void build() {
		vec3 eye = vec3(0.0f, 0.0f, 2.95f), vup = vec3(0, 1, 0), lookat = vec3(0, 0, 0);
		float fov = 45 * M_PI / 180;
		camera.set(eye, lookat, vup, fov);

		//La = vec3(0.4f, 0.4f, 0.4f);
		La = vec3(135.0f / 255.0f, 206.0f / 255.0f, 235.0f / 255.0f);
		vec3 lightDirection(1.0f, 1.0f, 1.0f), Le(2, 2, 2);
		lights.push_back(new Light(lightDirection, Le));

		genControlPoints(0.99f, 0.4232f);

		// nice blue kd(0.12f, 0.22f, 0.32f)
		vec3 kd1(0.32f, 0.12f, 0.12f), kd2(0.3f, 0.2f, 0.1f), kd3(0.1f, 0.3f, 0.2f), ks(2, 2, 2);
		Material* redRough = new RoughMaterial(kd1, ks, 50);
		Material* brownRough = new RoughMaterial(kd2, ks, 50);
		Material* greenRough = new RoughMaterial(kd3, ks, 50);

		Material* silverMirror = new ReflectiveMaterial(vec3(0.14f, 0.16f, 0.13f), vec3(4.1f, 2.6f, 3.1f));
		Material* goldenMirror = new ReflectiveMaterial(vec3(0.17f, 0.35f, 1.5f), vec3(3.1f, 2.7f, 1.9f));

		Ellipsoid* room = new Ellipsoid(3.0f, 1.0f, 3.0f, brownRough);
		room->makeRoom();
		objects.push_back(room);

		Ellipsoid* object1 = new Ellipsoid(0.2f, 0.37f, 0.23f, goldenMirror);
		object1->translate(vec3(0.0f, -0.8f, -0.1f));
		objects.push_back(object1);

		Cylinder* object2 = new Cylinder(0.1f, 0.2f, redRough);
		object2->translate(vec3(0.4f, -0.6f, 0.0f));
		objects.push_back(object2);

		Hyperboloid* object3 = new Hyperboloid(0.1f, 0.1f, 0.2f, greenRough);
		object3->rotate(90 * M_PI / 180.0f, vec3(1.0f, 0.0f, 0.0f));
		object3->translate(vec3(-0.5f, -0.6f, 0.0f));
		object3->setEnds(vec3(0.0f, -0.25f, 0.0f), vec3(0.0f, -0.95f, 0.0f));
		objects.push_back(object3);

		Hyperboloid* sunTube = new Hyperboloid(0.4232f, 0.4232f, 0.5f, silverMirror);
		sunTube->rotate(90 * M_PI / 180.0f, vec3(1.0f, 0.0f, 0.0f));
		sunTube->translate(vec3(0.0f, 0.99f, 0.0f));
		sunTube->setEnds(vec3(0.0f, 1.7f, 0.0f), vec3(0.0f, 0.99f, 0.0f));
		objects.push_back(sunTube);

		//for (int i = 0; i < 100; i++) {
		//	objects.push_back(new Sphere(vec3(rnd() - 0.5f, rnd() - 0.5f, rnd() - 0.5f), rnd() * 0.1f, silverMirror));
		//}
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
		for (Intersectable* object : objects) {
			Hit hit = object->intersect(ray); //  hit.t < 0 if no intersection
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
		}
		if (dot(ray.dir, bestHit.normal) > 0)
			bestHit.normal = bestHit.normal * (-1);
		return bestHit;
	}

	bool shadowIntersect(Ray ray) {	// for directional lights
		for (Intersectable* object : objects)
			if (object->intersect(ray).t > 0) return true;

		return false;
	}

	vec3 trace(Ray ray, int depth = 0) {
		if (depth > 5) return La;

		Hit hit = firstIntersect(ray);
		if (hit.t < 0)
			return La;

		vec3 outRadiance(0, 0, 0);

		if (hit.material->type == ROUGH) {
			outRadiance = hit.material->ka * La;

			for (Light* light : lights) {
				vec3 hitPlusEpsilon = hit.position + hit.normal * epsilon;
				Ray shadowRay(hitPlusEpsilon, light->direction);

				float cosTheta = dot(hit.normal, light->direction);
				if (cosTheta > 0 && !shadowIntersect(shadowRay)) {	// shadow computation
					outRadiance = outRadiance + light->Le * hit.material->kd * cosTheta;
					vec3 halfway = normalize(-ray.dir + light->direction);
					float cosDelta = dot(hit.normal, halfway);
					if (cosDelta > 0)
						outRadiance = outRadiance + light->Le * hit.material->ks * powf(cosDelta, hit.material->shininess);
				}

				//vec3 tubeRadiance(0, 0, 0);
				//for (vec3 cp : controlPoints) {
				//	tubeRadiance = tubeRadiance + trace(Ray(hitPlusEpsilon, cp - hitPlusEpsilon), depth + 1);
				//}
				//
				//outRadiance = outRadiance + tubeRadiance;
			}
		}

		if (hit.material->type == REFLECTIVE) {
			vec3 reflectedDir = ray.dir - hit.normal * dot(hit.normal, ray.dir) * 2.0f;
			float cosa = -dot(ray.dir, hit.normal);
			vec3 one(1, 1, 1);
			vec3 F = hit.material->F0 + (one - hit.material->F0) * pow(1 - cosa, 5);
			outRadiance = outRadiance + trace(Ray(hit.position + hit.normal * epsilon, reflectedDir), depth + 1) * F;
		}

		return outRadiance;
	}
};

GPUProgram gpuProgram; // vertex and fragment shaders
Scene scene;

// vertex shader in GLSL
const char* vertexSource = R"(
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
const char* fragmentSource = R"(
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

FullScreenTexturedQuad* fullScreenTexturedQuad;

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
