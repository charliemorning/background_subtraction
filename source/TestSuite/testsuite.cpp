#include "testsuite.h"


namespace TestSuite{

	Test::Test(std::ostream* osptr)
		: m_osptr(osptr),
		m_nPass(0),
		m_nFail(0)
	{
	}

	Test::~Test()
	{
	}


	long Test::getNumPassed() const
	{
		return m_nPass;
	}

	long Test::getNumFailed() const
	{
		return m_nFail;
	}

	const std::ostream* Test::getStream() const
	{
		return m_osptr;
	}

	void Test::setStream(std::ostream* osptr)
	{
		m_osptr = osptr;
	}

	void Test::succeed()
	{
		++m_nPass;
	}

	void Test::reset()
	{
		m_nPass = m_nFail = 0;
	}

	void Test::do_test(bool cond, const std::string &lbl, const char *fname, long lineno)
	{
		if(!cond)
			do_fail(lbl, fname, lineno);
		else
			succeed();
	}

	void Test::do_fail(const std::string &lbl, const char *fname, long lineno)
	{
		++m_nFail;
		if(m_osptr)
		{
			*m_osptr << typeid(*this).name()
				<< "failure: (" << lbl << ") ." << fname
				<< " (line " << lineno << ")\n";
		}
	}

	long Test::report() const
	{
		if(m_osptr)
		{
			*m_osptr << "Test \"" << typeid(*this).name()
				<< "\":\n\tPassed: " << m_nPass
				<< "\tFailed: " << m_nFail << "\n";
		}
		return m_nFail;
	}


	/************************************
	*implement of TestSuite class
	*************************************/

	std::string Suite::getName() const
	{
		return m_name;
	}

	const std::ostream* Suite::getStream() const
	{
		return m_osptr;
	}

	void Suite::setStream(std::ostream *osptr)
	{
		m_osptr = osptr;
	}

	size_t Suite::getTestsNum() const
	{
		return m_tests.size();
	}

	void Suite::addTest(TestSuite::Test *t) throw(TestSuiteError)
	{
		if(t == 0)
		{
			throw TestSuiteError("Null test in Suite::addTest");
		}
		else if(m_osptr && !t->getStream())
		{
			t->setStream(m_osptr);
		}
		m_tests.push_back(t);
		t->reset();
	}

	void Suite::addSuite(const TestSuite::Suite &s)
	{
		for(size_t i = 0; i < s.getTestsNum(); ++i)
		{
			assert(m_tests[i]);
			//addTest(s.
		}
	}

	void Suite::free()
	{
		for(size_t i = 0; i < m_tests.size(); ++i)
		{
			delete m_tests[i];
			m_tests[i] = NULL;
		}
	}

	void Suite::run()
	{
		reset();
		for(size_t i = 0; i < m_tests.size(); ++i)
		{
			assert(m_tests[i]);
			m_tests[i]->run();
		}

	}


	long Suite::report() const
	{
		if(m_osptr)
		{
			long totFail = 0;
			*m_osptr << "Suite \"" << m_name
				<< "\"\n======";

			size_t i;
			for(i = 0; i < m_tests.size(); ++i)
			{
				assert(m_tests[i]);
				totFail += m_tests[i]->report();
			}


			*m_osptr << "======";
			for(i = 0; i < m_tests.size(); ++i)
			{
				*m_osptr << '=';
			}
			*m_osptr << "=\n";
			return totFail;
		}
		else
			return getNumFailed();
	
	}

	long Suite::getNumPassed() const
	{
		long totPass = 0;

		for(size_t i = 0; i < m_tests.size(); ++i)
		{
			assert(m_tests[i]);
			totPass += m_tests[i]->getNumPassed();
		}
		return totPass;
	}

	long Suite::getNumFailed() const
	{
		long totFail = 0;

		for(size_t i = 0; i < m_tests.size(); ++i)
		{
			assert(m_tests[i]);
			totFail += m_tests[i]->getNumFailed();
		}
		return totFail;

	}

	void Suite::reset()
	{
		for(size_t i = 0; i < m_tests.size(); ++i)
		{
			assert(m_tests[i]);
			m_tests[i]->reset();
		}
	}
}