#ifndef _TEST_SUITE_H_
#define _TEST_SUITE_H_

#include <string>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <typeinfo>
#include <cassert>
#include <cstddef>

#define _TEST_(cond) \
	do_test(cond, #cond, __FILE__, __LINE__)
#define _FAIL_(str) \
	dofail(str, __FILE__, __LINE)

namespace TestSuite{
	class Test{
	public:

		Test(std::ostream* osptr = &std::cout);
		virtual ~Test();
		virtual void run() = 0;

		inline long getNumPassed() const;

		inline long getNumFailed() const;

		inline const std::ostream* getStream() const;

		inline void setStream(std::ostream* osptr);

		inline void succeed();

		long report() const;

		virtual inline void reset();


	protected:
		void do_test(bool cond, const std::string& lbl, const char* fname, long lineno);
		void do_fail(const std::string& lbl, const char* fname, long lineno);
	private:
		std::ostream* m_osptr;
		long m_nPass;
		long m_nFail;

		Test(const Test&);
		Test& operator=(const Test&);

	};

	class TestSuiteError: public std::logic_error{
	public:
		TestSuiteError(const std::string& s = "")
			: std::logic_error(s)
		{
		}
	};

	class Suite{
	public:
		Suite(const std::string& name, std::ostream* osptr = &std::cout)
			: m_name(name),
			m_osptr(osptr)
		{
		}

		inline std::string getName() const;

		long getNumPassed() const;
		long getNumFailed() const;

		inline const std::ostream* getStream() const;

		inline void setStream(std::ostream* osptr);

		inline size_t getTestsNum() const;

		void addTest(Test* t) throw(TestSuiteError);
		void addSuite(const Suite&);
		void run();
		long report() const;
		void free();

	protected:
	private:
		std::string m_name;
		std::ostream* m_osptr;
		std::vector<Test*> m_tests;
		void reset();
		Suite(const Suite&);
		Suite& operator=(const Suite&);
	};
}

#endif