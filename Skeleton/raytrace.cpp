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

// Source: video 8.3
struct RoughMaterial : Material {
	RoughMaterial(vec3 _kd, vec3 _ks, float _shininess) : Material(ROUGH) {
		ka = _kd * M_PI;
		kd = _kd;
		ks = _ks;
		shininess = _shininess;
	}
};

// Source: video 8.3
vec3 operator/(vec3 num, vec3 denom) {
	return vec3(num.x / denom.x, num.y / denom.y, num.z / denom.z);
}

// Source: video 8.3
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
	bool isRoom;

	// Source: Homework assignment video
	float f(vec4 r) {
		return dot(r * Q, r);
	}

	// Source: Homework assignment video
	vec3 gradf(vec4 r) {
		vec4 g = r * Q * 2;
		return vec3(g.x, g.y, g.z);
	}

	void makeRoom() {
		isRoom = true;
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

struct EllipticalCylinder : public Quadric {
	float a, b;
	vec3 top, bottom;

	EllipticalCylinder(float _a, float _b, Material* _material) {
		material = _material;
		a = _a; b = _b;
		Q = mat4(
			1.0f / (a * a), 0.0f, 0.0f, 0.0f,
			0.0f, 1.0f / (b * b), 0.0f, 0.0f,
			0.0f, 0.0f, 0.0f, 0.0f,
			0.0f, 0.0f, 0.0f, -1.0f
		);
	}

	void setEnds(vec3 _top, vec3 _bottom) {
		top = _top;
		bottom = _bottom;
	}

	Hit intersect(const Ray& ray) {
		Hit hit;
		vec3 nPos(0.0f, 0.0f, 1.0f), nNeg(0.0f, 0.0f, -1.0f);

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

struct Ellipsoid : public Quadric {
	float a, b, c;

	Ellipsoid(float _a, float _b, float _c, Material* _material) {
		a = _a; b = _b; c = _c;
		material = _material;
		Q = mat4(
			1.0f / (a * a), 0.0f, 0.0f, 0.0f,
			0.0f, 1.0f / (b * b), 0.0f, 0.0f,
			0.0f, 0.0f, 1.0f / (c * c), 0.0f,
			0.0f, 0.0f, 0.0f, -1.0f
		);
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
	vec3 top, bottom;

	EllipticalCone(float _a, float _b, float _c, Material* _material) {
		material = _material;
		a = _a; b = _b; c = _c; top = 0.0f;
		Q = mat4(
			1.0f / (a * a), 0.0f, 0.0f, 0.0f,
			0.0f, 1.0f / (b * b), 0.0f, 0.0f,
			0.0f, 0.0f, -1.0f / (c * c), 0.0f,
			0.0f, 0.0f, 0.0f, 0.0f
		);
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
	std::vector<Quadric*> objects;
	std::vector<Light*> lights;
	std::vector<vec3> controlPoints;
	Camera camera;
	vec3 La;
	float numOfControlPoints, A = 0.4232f * 0.4232f * M_PI;

public:
	void genControlPoints(float y, float r) {
		int cntGood = 0, cntBad = 0;
		for (int i = 0; i < numOfControlPoints; i++) {
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

		La = vec3(135.0f / 255.0f, 206.0f / 255.0f, 235.0f / 255.0f);
		vec3 lightDirection(2.0f, 1.7f, 2.0f), Le(100, 100, 100);
		lights.push_back(new Light(lightDirection, Le));

		numOfControlPoints = 100.0f;
		genControlPoints(0.99f, 0.4232f);

		vec3 kd1(0.32f, 0.12f, 0.12f), kd2(0.35f, 0.18f, 0.1f), kd3(0.1f, 0.3f, 0.2f), kd4(0.12f, 0.22f, 0.32f), ks(2, 2, 2);
		Material* redRough = new RoughMaterial(kd1, ks, 100);
		Material* brownRough = new RoughMaterial(kd2, ks, 80);
		Material* greenRough = new RoughMaterial(kd3, ks, 50);
		Material* blueRough = new RoughMaterial(kd4, ks, 50);

		Material* silverMirror = new ReflectiveMaterial(vec3(0.14f, 0.16f, 0.13f), vec3(4.1f, 2.6f, 3.1f));
		Material* goldenMirror = new ReflectiveMaterial(vec3(0.17f, 0.35f, 1.5f), vec3(3.1f, 2.7f, 1.9f));

		Ellipsoid* room = new Ellipsoid(3.0f, 1.0f, 3.0f, brownRough);
		room->makeRoom();
		objects.push_back(room);

		Ellipsoid* object1 = new Ellipsoid(0.5f, 0.8f, 0.7f, goldenMirror);
		object1->translate(vec3(0.5f, -0.5f, -1.3f));
		objects.push_back(object1);

		EllipticalCylinder* object2 = new EllipticalCylinder(0.25f, 0.15f, redRough);
		object2->translate(vec3(-0.4f, -0.55f, 0.0f));
		object2->setEnds(vec3(0.0f, 0.0f, -0.1f), vec3(0.0f, 0.0f, -1.5f));
		objects.push_back(object2);

		Hyperboloid* object3 = new Hyperboloid(0.15f, 0.15f, 0.18f, greenRough);
		object3->rotate(90 * M_PI / 180.0f, vec3(1.0f, 0.0f, 0.0f));
		object3->translate(vec3(-1.2f, -0.525f, -0.9f));
		object3->setEnds(vec3(0.0f, -0.25f, 0.0f), vec3(0.0f, -0.8f, 0.0f));
		objects.push_back(object3);

		EllipticalCone* object4 = new EllipticalCone(0.2f, 0.2f, 0.55f, blueRough);
		object4->rotate(90 * M_PI / 180.0f, vec3(1.0f, 0.0f, 0.0f));
		object4->translate(vec3(-0.83f, 0.17f, -1.6f));
		object4->setEnds(vec3(0.0f, 0.17f, 0.0f), vec3(0.f, -0.65f, 0.0f));
		objects.push_back(object4);

		Hyperboloid* sunTube = new Hyperboloid(0.4232f, 0.4232f, 0.5f, silverMirror);
		sunTube->rotate(90 * M_PI / 180.0f, vec3(1.0f, 0.0f, 0.0f));
		sunTube->translate(vec3(0.0f, 0.99f, 0.0f));
		sunTube->setEnds(vec3(0.0f, 1.7f, 0.0f), vec3(0.0f, 0.99f, 0.0f));
		objects.push_back(sunTube);

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

	Hit firstIntersect(Ray ray, bool isCPTrace) {
		Hit bestHit;
		for (Quadric* object : objects) {

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

	vec3 trace(Ray ray, bool isCPTrace = false, int depth = 0) {
		if (depth > 20) return La;

		Hit hit = firstIntersect(ray, isCPTrace);
		Light* sun = lights.at(0);
		if (hit.t < 0)
			return La + sun->Le * powf(dot(ray.dir, sun->direction), 10);

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

				vec3 tubeRadiance(0, 0, 0);
				for (vec3 cp : controlPoints) {
					vec3 cpRadiance(0, 0, 0);
					Ray shadowRay(hitPlusEpsilon, cp - hitPlusEpsilon);
					float cosTheta = dot(hit.normal, normalize(cp - hit.position));
					
					if (cosTheta > 0 && !shadowIntersect(shadowRay)) {
						// local illumination
						vec3 cpLe = trace(Ray(hitPlusEpsilon, cp - hit.position), true, depth + 1);
						cpRadiance = cpRadiance + cpLe * hit.material->kd * cosTheta;
				
						// Phong-Blinn
						vec3 halfway = normalize((-ray.dir + (cp - hit.position)) / 2.0f);
						float cosDelta = dot(hit.normal, halfway);
						if (cosDelta > 0) {
							cpRadiance = cpRadiance + cpLe * hit.material->ks * powf(cosDelta, hit.material->shininess);
						}
				
						// deltaOmega
						vec3 Li = hit.position - cp;
						float lengthLi = length(Li);
						float cosGamma = dot(normalize(Li), vec3(0, -1.0f, 0));
						float deltaOmega = (A / numOfControlPoints) * (cosGamma / (lengthLi * lengthLi));
						cpRadiance = cpRadiance * deltaOmega;
					}
					tubeRadiance = tubeRadiance + cpRadiance;
				}
				
				outRadiance = outRadiance + tubeRadiance;
			}
		}

		// Source: video 8.3
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
