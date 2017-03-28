/*
    This file is part of Mitsuba, a physically based rendering system.

    Copyright (c) 2007-2014 by Wenzel Jakob and others.

    Mitsuba is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License Version 3
    as published by the Free Software Foundation.

    Mitsuba is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#include <mitsuba/render/bsdf.h>
#include <mitsuba/hw/basicshader.h>
#include <mitsuba/core/warp.h>
#include "microfacet.h"
#include "rtrans.h"
#include "ior.h"


MTS_NAMESPACE_BEGIN


class Glints : public BSDF {
public:
	Glints(const Properties &props) : BSDF(props) {
		m_specularReflectance = new ConstantSpectrumTexture(
			props.getSpectrum("specularReflectance", Spectrum(1.0f)));
		m_diffuseReflectance = new ConstantSpectrumTexture(
			props.getSpectrum("diffuseReflectance", Spectrum(0.5f)));

		/* Specifies the internal index of refraction at the interface */
		Float intIOR = lookupIOR(props, "intIOR", "polypropylene");

		/* Specifies the external index of refraction at the interface */
		Float extIOR = lookupIOR(props, "extIOR", "air");

		if (intIOR < 0 || extIOR < 0 || intIOR == extIOR)
			Log(EError, "The interior and exterior indices of "
				"refraction must be positive and differ!");

		m_eta = intIOR / extIOR;

		m_nonlinear = props.getBoolean("nonlinear", false);

		MicrofacetDistribution distr(props);
		m_type = distr.getType();
		m_sampleVisible = distr.getSampleVisible();

		if (distr.isAnisotropic())
			Log(EError, "The 'roughplastic' plugin currently does not support "
				"anisotropic microfacet distributions!");

		m_alpha = new ConstantFloatTexture(distr.getAlpha());

		m_specularSamplingWeight = 0.0f;
	}

	Glints(Stream *stream, InstanceManager *manager)
	 : BSDF(stream, manager) {
		m_type = (MicrofacetDistribution::EType) stream->readUInt();
		m_sampleVisible = stream->readBool();
		m_specularReflectance = static_cast<Texture *>(manager->getInstance(stream));
		m_diffuseReflectance = static_cast<Texture *>(manager->getInstance(stream));
		m_glintReflectance = static_cast<Texture *>(manager->getInstance(stream));
		m_alpha = static_cast<Texture *>(manager->getInstance(stream));
		m_eta = stream->readFloat();
		m_nonlinear = stream->readBool();

		configure();
	}

	void serialize(Stream *stream, InstanceManager *manager) const {
		BSDF::serialize(stream, manager);

		stream->writeUInt((uint32_t) m_type);
		stream->writeBool(m_sampleVisible);
		manager->serialize(stream, m_specularReflectance.get());
		manager->serialize(stream, m_diffuseReflectance.get());
		manager->serialize(stream, m_alpha.get());
		stream->writeFloat(m_eta);
		stream->writeBool(m_nonlinear);
	}

	void configure() {
		bool constAlpha = m_alpha->isConstant();

		m_components.clear();

		m_components.push_back(EGlossyReflection | EFrontSide
			| ((constAlpha && m_specularReflectance->isConstant())
				? 0 : ESpatiallyVarying));
		m_components.push_back(EDiffuseReflection | EFrontSide
			| ((constAlpha && m_diffuseReflectance->isConstant())
				? 0 : ESpatiallyVarying));

		/* Verify the input parameters and fix them if necessary */
		m_specularReflectance = ensureEnergyConservation(
			m_specularReflectance, "specularReflectance", 1.0f);
		m_diffuseReflectance = ensureEnergyConservation(
			m_diffuseReflectance, "diffuseReflectance", 1.0f);

		/* Compute weights that further steer samples towards
		   the specular or diffuse components */
		Float dAvg = m_diffuseReflectance->getAverage().getLuminance(),
			  sAvg = m_specularReflectance->getAverage().getLuminance();
		m_specularSamplingWeight = sAvg / (dAvg + sAvg);

		m_invEta2 = 1.0f / (m_eta*m_eta);

		if (!m_externalRoughTransmittance.get()) {
			/* Load precomputed data used to compute the rough
			   transmittance through the dielectric interface */
			m_externalRoughTransmittance = new RoughTransmittance(m_type);

			m_externalRoughTransmittance->checkEta(m_eta);
			m_externalRoughTransmittance->checkAlpha(m_alpha->getMinimum().average());
			m_externalRoughTransmittance->checkAlpha(m_alpha->getMaximum().average());

			/* Reduce the rough transmittance data to a 2D slice */
			m_internalRoughTransmittance = m_externalRoughTransmittance->clone();
			m_externalRoughTransmittance->setEta(m_eta);
			m_internalRoughTransmittance->setEta(1/m_eta);

			/* If possible, even reduce it to a 1D slice */
			if (constAlpha)
				m_externalRoughTransmittance->setAlpha(
					m_alpha->eval(Intersection()).average());
		}

		m_usesRayDifferentials =
			m_specularReflectance->usesRayDifferentials() ||
			m_diffuseReflectance->usesRayDifferentials() ||
			m_glintReflectance->usesRayDifferentials() ||
			m_alpha->usesRayDifferentials();

		BSDF::configure();
	}

	Spectrum getDiffuseReflectance(const Intersection &its) const {
		/* Evaluate the roughness texture */
		Float alpha = m_alpha->eval(its).average();
		Float Ftr = m_externalRoughTransmittance->evalDiffuse(alpha);

		return m_diffuseReflectance->eval(its) * Ftr;
	}

	Spectrum getSpecularReflectance(const Intersection &its) const {
		return m_specularReflectance->eval(its);
	}

	/// Helper function: reflect \c wi with respect to a given surface normal
	inline Vector reflect(const Vector &wi, const Normal &m) const {
		return 2 * dot(wi, m) * Vector(m) - wi;
	}

	Spectrum eval(const BSDFSamplingRecord &bRec, EMeasure measure) const {
		bool hasSpecular = (bRec.typeMask & EGlossyReflection) &&
			(bRec.component == -1 || bRec.component == 0);
		bool hasDiffuse = (bRec.typeMask & EDiffuseReflection) &&
			(bRec.component == -1 || bRec.component == 1);

		if (measure != ESolidAngle ||
			Frame::cosTheta(bRec.wi) <= 0 ||
			Frame::cosTheta(bRec.wo) <= 0 ||
			(!hasSpecular && !hasDiffuse))
			return Spectrum(0.0f);

		/* Construct the microfacet distribution matching the
		   roughness values at the current surface position. */
		MicrofacetDistribution distr(
			m_type,
			m_alpha->eval(bRec.its).average(),
			m_sampleVisible
		);

		Spectrum result(0.0f);
		if (hasSpecular) {
			/* Calculate the reflection half-vector */
			const Vector H = normalize(bRec.wo+bRec.wi);

			/* Evaluate the microfacet normal distribution */
			const Float D = distr.eval(H);

			/* Fresnel term */
			const Float F = fresnelDielectricExt(dot(bRec.wi, H), m_eta);

			/* Smith's shadow-masking function */
			const Float G = distr.G(bRec.wi, bRec.wo, H);

			/* Calculate the specular reflection component */
			Float value = F * D * G /
				(4.0f * Frame::cosTheta(bRec.wi));

			result += m_specularReflectance->eval(bRec.its) * value;
		}

		if (hasDiffuse) {
			Spectrum diff = m_diffuseReflectance->eval(bRec.its);
			Float T12 = m_externalRoughTransmittance->eval(Frame::cosTheta(bRec.wi), distr.getAlpha());
			Float T21 = m_externalRoughTransmittance->eval(Frame::cosTheta(bRec.wo), distr.getAlpha());
			Float Fdr = 1-m_internalRoughTransmittance->evalDiffuse(distr.getAlpha());

			if (m_nonlinear)
				diff /= Spectrum(1.0f) - diff * Fdr;
			else
				diff /= 1-Fdr;

			result += diff * (INV_PI * Frame::cosTheta(bRec.wo) * T12 * T21 * m_invEta2);
		}

		return result;
	}

	Float pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const {
		bool hasSpecular = (bRec.typeMask & EGlossyReflection) &&
			(bRec.component == -1 || bRec.component == 0);
		bool hasDiffuse = (bRec.typeMask & EDiffuseReflection) &&
			(bRec.component == -1 || bRec.component == 1);

		if (measure != ESolidAngle ||
			Frame::cosTheta(bRec.wi) <= 0 ||
			Frame::cosTheta(bRec.wo) <= 0 ||
			(!hasSpecular && !hasDiffuse))
			return 0.0f;

		/* Construct the microfacet distribution matching the
		   roughness values at the current surface position. */
		MicrofacetDistribution distr(
			m_type,
			m_alpha->eval(bRec.its).average(),
			m_sampleVisible
		);

		/* Calculate the reflection half-vector */
		const Vector H = normalize(bRec.wo+bRec.wi);

		Float probDiffuse, probSpecular;
		if (hasSpecular && hasDiffuse) {
			/* Find the probability of sampling the specular component */
			probSpecular = 1-m_externalRoughTransmittance->eval(Frame::cosTheta(bRec.wi), distr.getAlpha());

			/* Reallocate samples */
			probSpecular = (probSpecular*m_specularSamplingWeight) /
				(probSpecular*m_specularSamplingWeight +
				(1-probSpecular) * (1-m_specularSamplingWeight));

			probDiffuse = 1 - probSpecular;
		} else {
			probDiffuse = probSpecular = 1.0f;
		}

		Float result = 0.0f;
		if (hasSpecular) {
			/* Jacobian of the half-direction mapping */
			const Float dwh_dwo = 1.0f / (4.0f * dot(bRec.wo, H));

			/* Evaluate the microfacet model sampling density function */
			const Float prob = distr.pdf(bRec.wi, H);

			result = prob * dwh_dwo * probSpecular;
		}

		if (hasDiffuse)
			result += probDiffuse * warp::squareToCosineHemispherePdf(bRec.wo);

		return result;
	}

	inline Spectrum sample(BSDFSamplingRecord &bRec, Float &_pdf, const Point2 &_sample) const {
		bool hasSpecular = (bRec.typeMask & EGlossyReflection) &&
			(bRec.component == -1 || bRec.component == 0);
		bool hasDiffuse = (bRec.typeMask & EDiffuseReflection) &&
			(bRec.component == -1 || bRec.component == 1);

		if (Frame::cosTheta(bRec.wi) <= 0 || (!hasSpecular && !hasDiffuse))
			return Spectrum(0.0f);

		bool choseSpecular = hasSpecular;
		Point2 sample(_sample);

		/* Construct the microfacet distribution matching the
		   roughness values at the current surface position. */
		MicrofacetDistribution distr(
			m_type,
			m_alpha->eval(bRec.its).average(),
			m_sampleVisible
		);

		Float probSpecular;
		if (hasSpecular && hasDiffuse) {
			/* Find the probability of sampling the specular component */
			probSpecular = 1 - m_externalRoughTransmittance->eval(Frame::cosTheta(bRec.wi), distr.getAlpha());

			/* Reallocate samples */
			probSpecular = (probSpecular*m_specularSamplingWeight) /
				(probSpecular*m_specularSamplingWeight +
				(1-probSpecular) * (1-m_specularSamplingWeight));

			if (sample.y < probSpecular) {
				sample.y /= probSpecular;
			} else {
				sample.y = (sample.y - probSpecular) / (1 - probSpecular);
				choseSpecular = false;
			}
		}

		if (choseSpecular) {
			/* Perfect specular reflection based on the microfacet normal */
			Normal m = distr.sample(bRec.wi, sample);
			bRec.wo = reflect(bRec.wi, m);
			bRec.sampledComponent = 0;
			bRec.sampledType = EGlossyReflection;

			/* Side check */
			if (Frame::cosTheta(bRec.wo) <= 0)
				return Spectrum(0.0f);
		} else {
			bRec.sampledComponent = 1;
			bRec.sampledType = EDiffuseReflection;
			bRec.wo = warp::squareToCosineHemisphere(sample);
		}
		bRec.eta = 1.0f;

		/* Guard against numerical imprecisions */
		_pdf = pdf(bRec, ESolidAngle);

		if (_pdf == 0)
			return Spectrum(0.0f);
		else
			return eval(bRec, ESolidAngle) / _pdf;
	}

	Spectrum sample(BSDFSamplingRecord &bRec, const Point2 &sample) const {
		Float pdf;
		return Glints::sample(bRec, pdf, sample);
	}

	void addChild(const std::string &name, ConfigurableObject *child) {
		if (child->getClass()->derivesFrom(MTS_CLASS(Texture))) {
			if (name == "alpha")
				m_alpha = static_cast<Texture *>(child);
			else if (name == "specularReflectance")
				m_specularReflectance = static_cast<Texture *>(child);
			else if (name == "diffuseReflectance")
				m_diffuseReflectance = static_cast<Texture *>(child);
			else
				BSDF::addChild(name, child);
		} else {
			BSDF::addChild(name, child);
		}
	}

	Float getRoughness(const Intersection &its, int component) const {
		Assert(component == 0 || component == 1);

		if (component == 0)
			return m_alpha->eval(its).average();
		else
			return std::numeric_limits<Float>::infinity();
	}

	std::string toString() const {
		std::ostringstream oss;
		oss << "Glints[" << endl
			<< "  id = \"" << getID() << "\"," << endl
			<< "  distribution = " << MicrofacetDistribution::distributionName(m_type) << "," << endl
			<< "  sampleVisible = " << m_sampleVisible << "," << endl
			<< "  alpha = " << indent(m_alpha->toString()) << "," << endl
			<< "  specularReflectance = " << indent(m_specularReflectance->toString()) << "," << endl
			<< "  diffuseReflectance = " << indent(m_diffuseReflectance->toString()) << "," << endl
			<< "  specularSamplingWeight = " << m_specularSamplingWeight << "," << endl
			<< "  diffuseSamplingWeight = " << (1-m_specularSamplingWeight) << "," << endl
			<< "  eta = " << m_eta << "," << endl
			<< "  nonlinear = " << m_nonlinear << endl
			<< "]";
		return oss.str();
	}

	Shader *createShader(Renderer *renderer) const;

	MTS_DECLARE_CLASS()
private:
	MicrofacetDistribution::EType m_type;
	ref<RoughTransmittance> m_externalRoughTransmittance;
	ref<RoughTransmittance> m_internalRoughTransmittance;
	ref<Texture> m_diffuseReflectance;
	ref<Texture> m_specularReflectance;
	ref<Texture> m_glintReflectance;
	ref<Texture> m_alpha;
	Float m_eta, m_invEta2;
	Float m_specularSamplingWeight;
	bool m_nonlinear;
	bool m_sampleVisible;
};

/**
 * GLSL port of the rough plastic shader. This version is much more
 * approximate -- it only supports the Beckmann distribution,
 * does everything in RGB, uses a cheaper shadowing-masking term, and
 * it also makes use of the Schlick approximation to the Fresnel
 * reflectance of dielectrics. When the roughness is lower than
 * \alpha < 0.2, the shader clamps it to 0.2 so that it will still perform
 * reasonably well in a VPL-based preview. There is no support for
 * non-linear effects due to internal scattering.
 */
class GlintsShader : public Shader {
public:
	GlintsShader(Renderer *renderer, const Texture *specularReflectance,
			const Texture *diffuseReflectance, const Texture *alpha, Float eta)
		: Shader(renderer, EBSDFShader),
			m_specularReflectance(specularReflectance),
			m_diffuseReflectance(diffuseReflectance),
			m_alpha(alpha) {
		m_specularReflectanceShader = renderer->registerShaderForResource(m_specularReflectance.get());
		m_diffuseReflectanceShader = renderer->registerShaderForResource(m_diffuseReflectance.get());
		m_alphaShader = renderer->registerShaderForResource(m_alpha.get());
		m_R0 = fresnelDielectricExt(1.0f, eta);
	}

	bool isComplete() const {
		return m_specularReflectanceShader.get() != NULL &&
			m_diffuseReflectanceShader.get() != NULL &&
			m_alphaShader.get() != NULL;
	}

	void putDependencies(std::vector<Shader *> &deps) {
		deps.push_back(m_specularReflectanceShader.get());
		deps.push_back(m_diffuseReflectanceShader.get());
		deps.push_back(m_alphaShader.get());
	}

	void cleanup(Renderer *renderer) {
		renderer->unregisterShaderForResource(m_specularReflectance.get());
		renderer->unregisterShaderForResource(m_diffuseReflectance.get());
		renderer->unregisterShaderForResource(m_alpha.get());
	}

	void resolve(const GPUProgram *program, const std::string &evalName, std::vector<int> &parameterIDs) const {
		parameterIDs.push_back(program->getParameterID(evalName + "_R0", false));
	}

	void bind(GPUProgram *program, const std::vector<int> &parameterIDs, int &textureUnitOffset) const {
		program->setParameter(parameterIDs[0], m_R0);
	}

	void generateCode(std::ostringstream &oss,
			const std::string &evalName,
			const std::vector<std::string> &depNames) const {
		oss << "uniform float " << evalName << "_R0;" << endl
			<< endl
			<< "float " << evalName << "_D(vec3 m, float alpha) {" << endl
			<< "    float ct = cosTheta(m);" << endl
			<< "    if (cosTheta(m) <= 0.0)" << endl
			<< "        return 0.0;" << endl
			<< "    float ex = tanTheta(m) / alpha;" << endl
			<< "    return exp(-(ex*ex)) / (pi * alpha * alpha *" << endl
			<< "               pow(cosTheta(m), 4.0));" << endl
			<< "}" << endl
			<< endl
			<< "float " << evalName << "_G(vec3 m, vec3 wi, vec3 wo) {" << endl
			<< "    if ((dot(wi, m) * cosTheta(wi)) <= 0 || " << endl
			<< "        (dot(wo, m) * cosTheta(wo)) <= 0)" << endl
			<< "        return 0.0;" << endl
			<< "    float nDotM = cosTheta(m);" << endl
			<< "    return min(1.0, min(" << endl
			<< "        abs(2 * nDotM * cosTheta(wo) / dot(wo, m))," << endl
			<< "        abs(2 * nDotM * cosTheta(wi) / dot(wi, m))));" << endl
			<< "}" << endl
			<< endl
			<< endl
			<< "float " << evalName << "_schlick(float ct) {" << endl
			<< "    float ctSqr = ct*ct, ct5 = ctSqr*ctSqr*ct;" << endl
			<< "    return " << evalName << "_R0 + (1.0 - " << evalName << "_R0) * ct5;" << endl
			<< "}" << endl
			<< endl
			<< "vec3 " << evalName << "(vec2 uv, vec3 wi, vec3 wo) {" << endl
			<< "    if (cosTheta(wi) <= 0 || cosTheta(wo) <= 0)" << endl
			<< "        return vec3(0.0);" << endl
			<< "    vec3 H = normalize(wi + wo);" << endl
			<< "    vec3 specRef = " << depNames[0] << "(uv);" << endl
			<< "    vec3 diffuseRef = " << depNames[1] << "(uv);" << endl
			<< "    float alpha = max(0.2, " << depNames[2] << "(uv)[0]);" << endl
			<< "    float D = " << evalName << "_D(H, alpha)" << ";" << endl
			<< "    float G = " << evalName << "_G(H, wi, wo);" << endl
			<< "    float F = " << evalName << "_schlick(1-dot(wi, H));" << endl
			<< "    return specRef    * (F * D * G / (4*cosTheta(wi))) + " << endl
			<< "           diffuseRef * ((1-F) * cosTheta(wo) * inv_pi);" << endl
			<< "}" << endl
			<< endl
			<< "vec3 " << evalName << "_diffuse(vec2 uv, vec3 wi, vec3 wo) {" << endl
			<< "    if (cosTheta(wi) < 0.0 || cosTheta(wo) < 0.0)" << endl
			<< "    	return vec3(0.0);" << endl
			<< "    vec3 diffuseRef = " << depNames[1] << "(uv);" << endl
			<< "    return diffuseRef * inv_pi * cosTheta(wo);"<< endl
			<< "}" << endl;
	}
	MTS_DECLARE_CLASS()
private:
	ref<const Texture> m_specularReflectance;
	ref<const Texture> m_diffuseReflectance;
	ref<const Texture> m_alpha;
	ref<Shader> m_specularReflectanceShader;
	ref<Shader> m_diffuseReflectanceShader;
	ref<Shader> m_alphaShader;
	Float m_R0;
};

Shader *Glints::createShader(Renderer *renderer) const {
	return new GlintsShader(renderer,
		m_specularReflectance.get(), m_diffuseReflectance.get(),
		m_alpha.get(), m_eta);
}

MTS_IMPLEMENT_CLASS(GlintsShader, false, Shader)
MTS_IMPLEMENT_CLASS_S(Glints, false, BSDF)
MTS_EXPORT_PLUGIN(Glints, "Glints BRDF");
MTS_NAMESPACE_END
