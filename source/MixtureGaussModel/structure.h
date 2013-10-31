#ifndef _STRUCTURE_H_
#define _STRUCTURE_H_

#include "globals.h"
#include "utility.h"

//_BEGIN_NAMESPACE_DETECT_

class GaussModel;

/*one gaussian model uint*/
class GaussModel
{
public:

	/*ctor*/
	GaussModel(const float& m = DEFUALT_MEAN,
		const float& sd = DEFUALT_STD_DEV, const float& w = DEFUALT_WEIGHT)
		: m_mean(m),
		m_stdDev(sd),
		m_weight(w)
	{
	}

	/*de-tor*/
	~GaussModel()
	{
	}

	float getMean() const
	{
		return m_mean;
	}

	float getStdDev() const
	{
		return m_stdDev;
	}

	float getWeight() const
	{
		return m_weight;
	}

	float getAlpha() const
	{
		return m_alpha;
	}

	float getValue() const
	{
		return m_value;
	}

	float getDifference() const
	{
		return m_diff;
	}

	void setMean(const float& mean)
	{
		m_mean = mean;
	}

	void setStdDev(const float& stdDev)
	{
		m_stdDev = stdDev;
	}

	void setWeight(const float& weight)
	{
		m_weight = weight;
	}

	void setAlpha(const float& alpha)
	{
		m_alpha = alpha;
	}

	void setValue(const float& value)
	{
		m_value = value;
	}

	void setDifference(const float& diff)
	{
		m_diff = diff;
	}

	/*increase mean when a component is matched*/
	inline void increaseMean(const float& alpha,
		const float& fg_val);

	/*increase standard deviation when a component is matched*/
	inline void increaseStdDev(const float& alpha,
		const float& fg_val);

	/*increase weight when a component is matched*/
	inline void increaseWeight(const float& alpha,
		const float& fg_val);

	/*increase weight when a component is matched*/
	inline void decreaseWeight(const float& alpha);

	

private:

	float m_mean;

	float m_stdDev;

	float m_weight;

	float m_p;

	static float m_alpha;

	float m_value;

	float m_diff;

	inline void _updateP(const float& alpha,
		const float& fg_val);

	inline float _gaussCalc(const float& fg_val);

};

float GaussModel::m_alpha = 0.01;

void GaussModel::increaseMean(const float& alpha,
										const float& fg_val)
{
	m_mean = (1 - m_p) * m_mean + m_p * fg_val;
}

void GaussModel::increaseStdDev(const float& alpha,
										  const float& fg_val)
{
	m_stdDev = sqrt((1.0 - m_p) * pow(m_stdDev, 2 ) +
		m_p * pow(fg_val - m_mean, 2));
}


void GaussModel::increaseWeight(const float& alpha,
										  const float& fg_val)
{
	m_weight = (1 - alpha) * m_weight + alpha;
	_updateP(alpha, fg_val);
}


void GaussModel::decreaseWeight(const float& alpha)
{
	m_weight = (1 - alpha) * m_weight;
}


float GaussModel::_gaussCalc(const float& fg_val)
{
	return 1 / (pow(2 * 3.141, 1.5) * m_stdDev )  *
		exp( -0.5 * pow((double)(fg_val - m_stdDev), 2.0) / pow((double)m_stdDev, 2.0));
}


void GaussModel::_updateP(const float& alpha,
									const float& fg_val)
{
	//m_p = alpha / m_weight;
	m_p = alpha * m_weight * _gaussCalc(fg_val);
}

template <typename DataType>
void abs(DataType& data)
{
	if(data < 0)
		data = -data;
}


int compare_gauss_model(const void* lv, const void* rv)
{
	
	float left = static_cast<const GaussModel*>(lv)->getWeight() /
		static_cast<const GaussModel*>(lv)->getStdDev();

	float right = static_cast<const GaussModel*>(lv)->getWeight() /
		static_cast<const GaussModel*>(lv)->getStdDev();

	if(left < right)
	{
		return -1;
	}
	else if(left == right)
	{
		return 0;
	}
	else
	{
		return 1;
	}
}




template <unsigned int GaussNum = 3>
class Gaussian
{
public:

	Gaussian(const unsigned int& width,
		const unsigned int& height);

	~Gaussian();

	void initiate(const cv::Mat& frame);

	void process(const cv::Mat& frame, cv::Mat& foreground);

	void updateBackground(bool match);

private:

	void _init();

	void _matchUpdateModel(GaussModel& model,
		const float& frVal, const float& alpha);

	void _unmatchUpdateModel(GaussModel& model,
		const float& frVal, const float& alpha);

	/*normalize the weight of each gaussian model*/
	void _normalize(const int idx);

	void _resetModel(GaussModel& model, const float& frVal);

	GaussModel (*m_model)[GaussNum];

	unsigned int m_width;

	unsigned int m_height;

};

template <unsigned int GaussNum>
Gaussian<GaussNum>::Gaussian(const unsigned int& width,
									   const unsigned int& height)
	: m_width(width),
	m_height(height)
{
	_init();
}

template <unsigned int GaussNum>
Gaussian<GaussNum>::~Gaussian()
{
}

template <unsigned int GaussNum>
void Gaussian<GaussNum>::initiate(const cv::Mat& frame)
{

	cv::MatConstIterator_<cv::Vec3b> frame_iter;

	int nChannel = 3;
	int cnt = 0;
	frame_iter = frame.begin<cv::Vec3b>();

	

	while(frame_iter != frame.end<cv::Vec3b>())
	{
		for(int i = 0; i < nChannel; ++i)
		{
			uchar t = (*frame_iter)[i];
			m_model[cnt][i].setMean((*frame_iter)[i]);
		}
		cnt++;
		frame_iter++;

	}

}


template <unsigned int GaussNum>
void Gaussian<GaussNum>::_init()
{
	for(uint i = 0; i < GaussNum; i++)
	{
		m_model = new GaussModel [m_width * m_height] [GaussNum];
	}
}


template <unsigned int GaussNum>
void Gaussian<GaussNum>::_matchUpdateModel(
	GaussModel& model, const float& frVal, const float& alpha)
{
	model.increaseWeight(alpha, frVal);
	model.increaseMean(alpha, frVal);
	model.increaseStdDev(alpha, frVal);

}

template <unsigned int GaussNum>
void Gaussian<GaussNum>::_unmatchUpdateModel(
	GaussModel& model, const float &frVal, const float &alpha)
{
	model.decreaseWeight(alpha);
}

template <unsigned int GaussNum>
void Gaussian<GaussNum>::_resetModel(
									 GaussModel& model, const float& frVal)
{
	model.setMean(frVal);
	model.setStdDev(DEFUALT_STD_DEV);
}

template <unsigned int GaussNum>
void Gaussian<GaussNum>::_normalize(const int idx)
{
	float weight_sum = 0;
	for(int i = 0; i < GaussNum; ++i)
	{
		weight_sum += m_model[idx][i].getWeight();
	}

	for(int i = 0; i < GaussNum; ++i)
	{
		float temp_weight = m_model[idx][i].getWeight() / weight_sum;
		m_model[idx][i].setWeight(temp_weight);
	}
}


template <unsigned int GaussNum>
void Gaussian<GaussNum>::process(const cv::Mat& frame, cv::Mat& foreground)
{

	const size_t nChannel = 3;

	cv::MatConstIterator_<cv::Vec3b> frame_iter = frame.begin<cv::Vec3b>();
	cv::MatIterator_<uchar> foreground_iter;

	foreground_iter = foreground.begin<uchar>();

	size_t cnt = 0;

	float alpha = 0.01;
	while(frame_iter != frame.end<cv::Vec3b>())
	{
		size_t match = 0;


		for(size_t i = 0; i < nChannel; ++i)
		{

			m_model[cnt][i].setValue((*frame_iter)[i]);
	

			m_model[cnt][i].setDifference(abs<float>(m_model[cnt][i].getMean() - m_model[cnt][i].getValue()));

			if(m_model[cnt][i].getDifference() <= 4 * m_model[cnt][i].getStdDev())
			{
				match = i;

				_matchUpdateModel(m_model[cnt][i], m_model[cnt][i].getValue(), alpha);

			}
			else
			{
				_unmatchUpdateModel(m_model[cnt][i], m_model[cnt][i].getValue(), alpha);
			}
		}

		_normalize(cnt);

		//no match
		if(match == 0)
		{
			qsort((void*)(m_model[cnt]), GaussNum, sizeof(GaussModel), compare_gauss_model);
			_resetModel(m_model[cnt][0], m_model[cnt][0].getValue());
		}
		
		if(match)
		{
			*foreground_iter = 0;
		}
		else
		{
			*foreground_iter = 255;
		}

		cnt++;
		foreground_iter++;
		frame_iter++;

	}

}









//_END_NAMESPACE_DETECT_

#endif