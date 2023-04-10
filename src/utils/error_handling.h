#ifndef ERROR_HANDLING_CHOLLA_H
#define ERROR_HANDLING_CHOLLA_H
#include <stdlib.h>

#include "../global/global.h"
void chexit(int code);

/*!
 * \brief Check that the Cholla configuration and parameters don't have any significant errors. Mostly compile time
 * checks.
 *
 */
void Check_Configuration(parameters const &P);
#endif /*ERROR_HANDLING_CHOLLA_H*/
